#ifndef CU_ZFP_VARIABLE_CUH
#define CU_ZFP_VARIABLE_CUH

#include "shared.h"

#include <cub/cub.cuh>

#include <cooperative_groups.h> // Requires CUDA >= 9
namespace cg = cooperative_groups;

namespace cuZFP
{

    // *******************************************************************************

    // Copy a chunk of 16-bit stream lengths into the 64-bit offsets array
    // to compute prefix sums. The first value in offsets is the "base" of the prefix sum
    __global__ void copy_length(ushort *length,
                                unsigned long long *offsets,
                                unsigned long long first_stream,
                                int nstreams_chunk)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= nstreams_chunk)
            return;
        offsets[index + 1] = length[first_stream + index];
    }

    void copy_length_launch(ushort *bitlengths,
                            unsigned long long *chunk_offsets,
                            unsigned long long first,
                            int nstreams_chunk)
    {
        dim3 blocks((nstreams_chunk - 1) / 1024 + 1, 1, 1);
        copy_length<<<blocks, 1024>>>(bitlengths, chunk_offsets, first, nstreams_chunk);
    }

    // *******************************************************************************

    // Each tile loads the compressed but uncompacted data to shared memory.
    // Input alignment can be anything (1-bit) as maxbits is not always a multiple of 8,
    // so the data is aligned on the fly (first bit of the bitstream on bit 0 in shared memory)
    template <uint tile_size>
    __device__ inline void load_to_shared(const uint *streams,                   // Input data
                                          uint *sm,                              // Shared memory
                                          const unsigned long long &offset_bits, // Offset in bits for the stream
                                          const uint &length_bits,               // Length in bits for this stream
                                          const int &maxpad32)                   // Next multiple of 32 of maxbits
    {
        uint misaligned = offset_bits & 31;
        unsigned long long offset_32 = offset_bits / 32;
        for (int i = threadIdx.x; i * 32 < length_bits; i += tile_size)
        {
            // Align even if already aligned
            uint low = streams[offset_32 + i];
            uint high = 0;
            if ((i + 1) * 32 < misaligned + length_bits)
                high = streams[offset_32 + i + 1];
            sm[threadIdx.y * maxpad32 + i] = __funnelshift_r(low, high, misaligned);
        }
    }

    // Read the input bitstreams from shared memory, align them relative to the
    // final output alignment, compact all the aligned bitstreams in sm_out,
    // then write all the data (coalesced) to global memory, using atomics only
    // for the first and last elements
    template <int tile_size, int num_tiles>
    __device__ inline void process(bool valid_stream,
                                   unsigned long long &offset0,     // Offset in bits of the first bitstream of the block
                                   const unsigned long long offset, // Offset in bits for this stream
                                   const int &length_bits,          // length of this stream
                                   const int &add_padding,          // padding at the end of the block, in bits
                                   const int &tid,                  // global thread index inside the thread block
                                   uint *sm_in,                     // shared memory containing the compressed input data
                                   uint *sm_out,                    // shared memory to stage the compacted compressed data
                                   uint maxpad32,                   // Leading dimension of the shared memory (padded maxbits)
                                   uint *sm_length,                 // shared memory to compute a prefix-sum inside the block
                                   uint *output)                    // output pointer
    {
        // All streams in the block will align themselves on the first stream of the block
        int misaligned0 = offset0 & 31;
        int misaligned = offset & 31;
        int off_smin = threadIdx.y * maxpad32;
        int off_smout = ((int)(offset - offset0) + misaligned0) / 32;
        offset0 /= 32;

        if (valid_stream)
        {
            // Loop on the whole bitstream (including misalignment), 32 bits per thread
            for (int i = threadIdx.x; i * 32 < misaligned + length_bits; i += tile_size)
            {
                // Merge 2 values to create an aligned value
                uint v0 = i > 0 ? sm_in[off_smin + i - 1] : 0;
                uint v1 = sm_in[off_smin + i];
                v1 = __funnelshift_l(v0, v1, misaligned);

                // Mask out neighbor bitstreams
                uint mask = 0xffffffff;
                if (i == 0)
                    mask &= 0xffffffff << misaligned;
                if ((i + 1) * 32 > misaligned + length_bits)
                    mask &= ~(0xffffffff << ((misaligned + length_bits) & 31));
                
                atomicAdd(sm_out + off_smout + i, v1 & mask);
            }
        }

        // First thread working on each bistream writes the length in shared memory
        // Add zero-padding bits if needed (last bitstream of last chunk)
        // The extra bits in shared mempory are already zeroed.
        if (threadIdx.x == 0)
            sm_length[threadIdx.y] = length_bits + add_padding;

        // This synchthreads protects sm_out and sm_length.
        __syncthreads();

        // Compute total length for the threadblock
        uint total_length = 0;
        for (int i = tid & 31; i < num_tiles; i += 32)
            total_length += sm_length[i];
        for (int i = 1; i < 32; i *= 2)
            total_length += __shfl_xor_sync(0xffffffff, total_length, i);

        // Write the shared memory output data to global memory, using all the threads
        for (int i = tid; i * 32 < misaligned0 + total_length; i += tile_size * num_tiles)
        {
            // Mask out the beginning and end of the block if unaligned
            uint mask = 0xffffffff;
            if (i == 0)
                mask &= 0xffffffff << misaligned0;
            if ((i + 1) * 32 > misaligned0 + total_length)
                mask &= ~(0xffffffff << ((misaligned0 + total_length) & 31));
            // Reset the shared memory to zero for the next iteration.
            uint value = sm_out[i];
            sm_out[i] = 0;
            // Write to global memory. Use atomicCAS for partially masked values
            // Working in-place, the output buffer has not been memset to zero
            if (mask == 0xffffffff)
                output[offset0 + i] = value;
            else
            {
                uint assumed, old = output[offset0 + i];
                do
                {
                    assumed = old;
                    old = atomicCAS(output + offset0 + i, assumed, (assumed & ~mask) + (value & mask));
                } while (assumed != old);
            }
        }
    }

    // In-place bitstream concatenation: compacting blocks containing different number
    // of bits, with the input blocks stored in bins of the same size
    // Using a 2D tile of threads,
    // threadIdx.y = Index of the stream
    // threadIdx.x = Threads working on the same stream
    // Must launch dim3(tile_size, num_tiles, 1) threads per block.
    // Offsets has a length of (nstreams_chunk + 1), offsets[0] is the offset in bits
    // where stream 0 starts, it must be memset to zero before launching the very first chunk,
    // and is updated at the end of this kernel.
    template <int tile_size, int num_tiles>
    __launch_bounds__(tile_size *num_tiles)
        __global__ void concat_bitstreams_chunk(uint *__restrict__ streams,
                                                unsigned long long *__restrict__ offsets,
                                                unsigned long long first_stream_chunk,
                                                int nstreams_chunk,
                                                bool last_chunk,
                                                int maxbits,
                                                int maxpad32)
    {
        cg::grid_group grid = cg::this_grid();
        __shared__ uint sm_length[num_tiles];
        extern __shared__ uint sm_in[];              // sm_in[num_tiles * maxpad32]
        uint *sm_out = sm_in + num_tiles * maxpad32; // sm_out[num_tiles * maxpad32 + 2]
        int tid = threadIdx.y * tile_size + threadIdx.x;
        int grid_stride = gridDim.x * num_tiles;
        int first_bitstream_block = blockIdx.x * num_tiles;
        int my_stream = first_bitstream_block + threadIdx.y;

        // Zero the output shared memory. Will be reset again inside process().
        for (int i = tid; i < num_tiles * maxpad32 + 2; i += tile_size * num_tiles)
            sm_out[i] = 0;

        // Loop on all the bitstreams of the current chunk, using the whole resident grid.
        // All threads must enter this loop, as they have to synchronize inside.
        for (int i = 0; i < nstreams_chunk; i += grid_stride)
        {
            bool valid_stream = my_stream + i < nstreams_chunk;
            unsigned long long offset0 = 0;
            unsigned long long offset = 0;
            uint length_bits = 0;
            uint add_padding = 0;
            if (valid_stream)
            {
                offset0 = offsets[first_bitstream_block + i];
                offset = offsets[my_stream + i];
                unsigned long long offset_bits = (first_stream_chunk + my_stream + i) * maxbits;
                unsigned long long next_offset_bits = offsets[my_stream + i + 1];
                length_bits = (uint)(next_offset_bits - offset);
                load_to_shared<tile_size>(streams, sm_in, offset_bits, length_bits, maxpad32);
                if (last_chunk && (my_stream + i == nstreams_chunk - 1))
                {
                    uint partial = next_offset_bits & 63;
                    add_padding = (64 - partial) & 63;
                }
            }

            // Check if there is overlap between input and output at the grid level.
            // Grid sync if needed, otherwise just syncthreads to protect the shared memory.
            int last_stream = min(nstreams_chunk, i + grid_stride);
            unsigned long long writing_to = (offsets[last_stream] + 31) / 32;
            unsigned long long reading_from = (first_stream_chunk + i) * maxbits;
            if (writing_to >= reading_from)
                grid.sync();
            else
                __syncthreads();

            // Compact the shared memory data and write it to global memory
            process<tile_size, num_tiles>(valid_stream, offset0, offset, length_bits, add_padding,
                                          tid, sm_in, sm_out, maxpad32, sm_length, streams);
        }

        // Reset the base of the offsets array, for the next chunk's prefix sum
        if (blockIdx.x == 0 && tid == 0)
            offsets[0] = offsets[nstreams_chunk];
    }

    void chunk_process_launch(uint *streams,
                              unsigned long long *chunk_offsets,
                              unsigned long long first,
                              int nstream_chunk,
                              bool last_chunk,
                              int nbitsmax,
                              int num_sm)
    {
        int maxpad32 = (nbitsmax + 31) / 32;
        void *kernelArgs[] = {(void *)&streams,
                              (void *)&chunk_offsets,
                              (void *)&first,
                              (void *)&nstream_chunk,
                              (void *)&last_chunk,
                              (void *)&nbitsmax,
                              (void *)&maxpad32};
        // Increase the number of threads per ZFP block ("tile") as nbitsmax increases
        // Compromise between coalescing, inactive threads and shared memory size <= 48KB
        // Total shared memory used = (2 * num_tiles * maxpad + 2) x 32-bit dynamic shared memory
        // and num_tiles x 32-bit static shared memory.
        // The extra 2 elements of dynamic shared memory are needed to handle unaligned output data
        // and potential zero-padding to the next multiple of 64 bits.
        // Block sizes set so that the shared memory stays < 48KB.
        int max_blocks = 0;
        if (nbitsmax <= 352)
        {
            constexpr int tile_size = 1;
            constexpr int num_tiles = 512;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                          concat_bitstreams_chunk<tile_size, num_tiles>,
                                                          tile_size * num_tiles, shmem);
            max_blocks *= num_sm;
            dim3 threads(tile_size, num_tiles, 1);
            cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                                        dim3(max_blocks, 1, 1), threads, kernelArgs, shmem, 0);
        }
        else if (nbitsmax <= 1504)
        {
            constexpr int tile_size = 4;
            constexpr int num_tiles = 128;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                          concat_bitstreams_chunk<tile_size, num_tiles>,
                                                          tile_size * num_tiles, shmem);
            max_blocks *= num_sm;
            dim3 threads(tile_size, num_tiles, 1);
            cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                                        dim3(max_blocks, 1, 1), threads, kernelArgs, shmem, 0);
        }
        else if (nbitsmax <= 6112)
        {
            constexpr int tile_size = 16;
            constexpr int num_tiles = 32;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                          concat_bitstreams_chunk<tile_size, num_tiles>,
                                                          tile_size * num_tiles, shmem);
            max_blocks *= num_sm;
            dim3 threads(tile_size, num_tiles, 1);
            cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                                        dim3(max_blocks, 1, 1), threads, kernelArgs, shmem, 0);
        }
        else // Up to 24512 bits, so works even for largest 4D.
        {
            constexpr int tile_size = 64;
            constexpr int num_tiles = 8;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                          concat_bitstreams_chunk<tile_size, num_tiles>,
                                                          tile_size * num_tiles, shmem);
            max_blocks *= num_sm;
            dim3 threads(tile_size, num_tiles, 1);
            cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                                        dim3(max_blocks, 1, 1), threads, kernelArgs, shmem, 0);
        }
    }

    // *******************************************************************************

} // namespace cuZFP
#endif
