#ifndef ULL128_H
#define ULL128_H
//
//struct Bitter
//{
//	unsigned long long int x;
//	unsigned int y;
//};

typedef  ulonglong2 Bitter;

__host__ __device__
Bitter make_bitter(unsigned long long int in0, unsigned int in1){
	//Bitter ret;
	//ret.x = in0;
	//ret.y = in1;
	//return ret;
	return make_ulonglong2(in0, in1);
}

__device__ __host__
Bitter lshiftull2(const Bitter &in, size_t len)
{

	Bitter a = in;
	if (len > 0){
		unsigned long long value = a.x;
		if (len < 64){
			unsigned long long v = value >> (64 - len);
			a.y <<= len;
			a.y += v;

			a.x <<= len;
		}
		else{
			a.y = a.x = 0;

			len -= 64;
			unsigned long long v = value << len;
			a.y += v;
		}
	}
	return a;
}
__device__ __host__
Bitter rshiftull2(const Bitter &in, size_t len)
{
	Bitter a = in;
	unsigned long long value = a.y;
	if (len < 64){
		a.x >>= len;
		value <<= (64 - len);
		a.x += value;

		a.y >>= len;
	}
	else{
		a.y >>= (len - 64);
		a.x = a.y;
		a.y = 0;
	}



	return a;
}

__device__ __host__
Bitter subull2(Bitter in1, Bitter in2)
{
	Bitter difference;
	difference.y = in1.y - in2.y;
	difference.x = in1.x - in2.x;
	// check for underflow of low 64 bits, subtract carry to high
	if (difference.y > in1.x)
		--difference.y;
	return difference;
}



#endif