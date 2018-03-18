#include "gtest/gtest.h"

// assumes a constants/<dim><type>.h is already included

class ArrayNdTestFixture : public ::testing::TestWithParam<int> {
protected:
  virtual void SetUp() {
    bitstreamChecksums_[0] = CHECKSUM_FR_8_COMPRESSED_BITSTREAM;
    bitstreamChecksums_[1] = CHECKSUM_FR_16_COMPRESSED_BITSTREAM;
    bitstreamChecksums_[2] = CHECKSUM_FR_32_COMPRESSED_BITSTREAM;

    decompressedChecksums_[0] = CHECKSUM_FR_8_DECOMPRESSED_ARRAY;
    decompressedChecksums_[1] = CHECKSUM_FR_16_DECOMPRESSED_ARRAY;
    decompressedChecksums_[2] = CHECKSUM_FR_32_DECOMPRESSED_ARRAY;
  }

  double getRate() { return 1u << (GetParam() + 3); }

  uint64 getExpectedBitstreamChecksum() { return bitstreamChecksums_[GetParam()]; }

  uint64 getExpectedDecompressedChecksum() { return decompressedChecksums_[GetParam()]; }

  uint64 bitstreamChecksums_[3], decompressedChecksums_[3];
};
