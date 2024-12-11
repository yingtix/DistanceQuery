#include "CudaBase/helper_math.h"

#ifndef USE_14Dop
//#define USE_14Dop
#endif // !USE_14Dop

#ifdef USE_14Dop
__host__ __device__ struct g_box
{
	float3 _min, _max;
	float4 minCorner;
	float4 maxCorner;
	__host__ __device__ inline void add(const g_box& b)
	{
		_min = fminf(_min, b._min);
		_max = fmaxf(_max, b._max);
		minCorner = fminf(minCorner, b.minCorner);
		maxCorner = fmaxf(maxCorner, b.maxCorner);
	}

	__host__ __device__ inline void add(const float3& v)
	{
		_min = fminf(_min, v);
		_max = fmaxf(_max, v);
		// 1 1 1
		// 1 1 -1
		// 1 -1 1
		// 1 -1 -1
		float4 temp = make_float4(v.x + v.y + v.z, v.x + v.y - v.z, v.x - v.y + v.z, v.x - v.y - v.z);
		minCorner = fminf(minCorner, temp);
		maxCorner = fmaxf(maxCorner, temp);
	}

	__host__ __device__ inline void set(const float3& v)
	{
		_min = v;
		_max = v;
		minCorner = maxCorner = make_float4(v.x + v.y + v.z, v.x + v.y - v.z, v.x - v.y + v.z, v.x - v.y - v.z);
	}

	__host__ __device__ inline float3 center()
	{
		return (_min + _max) * 0.5;
	}
};
#else
__host__ __device__ struct g_box {
	float3 _min, _max;
	__host__ __device__  inline void add(const g_box& b)
	{
		_min = fminf(_min, b._min);
		_max = fmaxf(_max, b._max);
	}

	__host__ __device__ inline void add(const float3& v)
	{
		_min = fminf(_min, v);
		_max = fmaxf(_max, v);
	}

	__host__ __device__ inline void set(const float3& v)
	{
		_min = v;
		_max = v;
	}

	__host__ __device__ inline float3 center()
	{
		return (_min + _max) * 0.5;
	}
};
#endif //

__host__ __device__ struct g_transf {
	float3 _off;
	float3x3 _trf;

	__host__ __device__ inline float3 apply(const float3& v) const {
		return _trf * v + _off;
	}

	__host__ __device__ inline g_transf inverse() const {
		g_transf ret;
		ret._trf = getTrans(_trf);
		ret._off = ret._trf * (_off * -1);
		return ret;
	}
};

__host__ __device__ struct g_bvtt_SoA {
	int* a;
	int* b;
	float* min;
};

#include <vector>

struct DistanceResult{
	float min;
	int lab;
	float sum_time;
	int oldId1, oldId2;
	int id1, id2;
	std::vector<float> times;
	std::vector<int> proDeeps;
	std::vector<int> bvtts;
};
