/**
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
*  This file implements common mathematical operations on vector types
*  (float3, float4 etc.) since these are not provided as standard by CUDA.
*
*  The syntax is modeled on the Cg standard library.
*
*  This is part of the Helper library includes
*
*    Thanks to Linh Hah for additions and fixes.
*/

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include "cuda_runtime.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
	return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline int max(int a, int b)
{
	return a > b ? a : b;
}

inline int min(int a, int b)
{
	return a < b ? a : b;
}

inline float rsqrtf(float x)
{
	return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
	return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
	return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
	return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
	return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
	return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
	return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
	return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
	return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
	return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
	return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
	return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
	return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
	return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
	return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
	return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
	return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
	return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
	return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
	return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
	return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
	return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
	return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
	return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
	return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
	return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
	return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
	return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
	return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
	return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
	return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
	return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
	return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
	return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
	return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2& a)
{
	return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2& a)
{
	return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3& a)
{
	return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4& a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4& a)
{
	return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2& a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
	return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
	return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2& a, float b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2& a, int2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
	return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
	return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2& a, int b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
	return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2& a, uint2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
	return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
	return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2& a, uint b)
{
	a.x += b;
	a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3& a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3& a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3& a, int3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(int3& a, int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3& a, uint3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3& a, uint b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4& a, float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4& a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
	return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(int4& a, int4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
	return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
	return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(int4& a, int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4& a, uint4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
	return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
	return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(uint4& a, uint b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2& a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
	return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
	return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2& a, float b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2& a, int2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
	return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
	return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2& a, int b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
	return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2& a, uint2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
	return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
	return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2& a, uint b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3& a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
	return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3& a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3& a, int3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
	return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
	return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3& a, int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
	return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3& a, uint3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
	return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
	return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3& a, uint b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4& a, float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(float4& a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
	return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(int4& a, int4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
	return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
	return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4& a, int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
	return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4& a, uint4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
	return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
	return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4& a, uint b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2& a, float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2& a, float b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2& a, int2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
	return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
	return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2& a, int b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
	return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2& a, uint2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
	return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
	return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2& a, uint b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3& a, float3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3& a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3& a, int3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
	return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
	return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3& a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
	return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3& a, uint3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
	return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
	return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3& a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4& a, float4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4& a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
	return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(int4& a, int4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
	return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
	return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4& a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
	return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4& a, uint4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
	return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
	return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4& a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2& a, float2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
	return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2& a, float b)
{
	a.x /= b;
	a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
	return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3& a, float3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3& a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
	return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float4& a, float4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float4& a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
	return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
	return make_int2(min(a.x, b.x), min(a.y, b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
	return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
	return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
	return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
	return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
	return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
	return make_int2(max(a.x, b.x), max(a.y, b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
	return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
	return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
	return make_uint2(max(a.x, b.x), max(a.y, b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
	return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
	return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
	return a + t * (b - a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
	return a + t * (b - a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
	return a + t * (b - a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
	return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
	return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
	return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
	return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
	return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
	return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
	return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
	return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
	return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
	return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
	return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
	return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
	return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
	return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
	return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
	return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
	return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
	return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
	return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
	return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float length2(float3 v) {
	return dot(v, v);
}
inline __host__ __device__ float length(float4 v)
{
	return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
	return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
	return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
	return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
	return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
	return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
	return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
	return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
	return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
	return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
	return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
	return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
	return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
	return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
	return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
	return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
	return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
	return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
	float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y * y * (3.0f - (2.0f * y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
	float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y * y * (make_float2(3.0f) - (make_float2(2.0f) * y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
	float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
	float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y * y * (make_float4(3.0f) - (make_float4(2.0f) * y)));
}


typedef struct {
	float c[6];

	inline __host__ __device__ float3 c0() const { return make_float3(c[0], c[1], c[2]); }
	inline __host__ __device__ float3 c1() const { return make_float3(c[3], c[4], c[5]); }
} float3x2;

typedef struct {
	float c[6];

	inline __host__ __device__ float2 c0() const { return make_float2(c[0], c[1]); }
	inline __host__ __device__ float2 c1() const { return make_float2(c[2], c[3]); }
	inline __host__ __device__ float2 c2() const { return make_float2(c[4], c[5]); }
} float2x3;

typedef struct {
	float c[4];

	inline __host__ __device__ float2 c0() const { return make_float2(c[0], c[1]); }
	inline __host__ __device__ float2 c1() const { return make_float2(c[2], c[3]); }
} float2x2;

inline __host__ __device__ float3x2 make_float3x2(float3 c0, float3 c1)
{
	float3x2 t;

	t.c[0] = c0.x;
	t.c[1] = c0.y;
	t.c[2] = c0.z;
	t.c[3] = c1.x;
	t.c[4] = c1.y;
	t.c[5] = c1.z;
	return t;
}

inline __host__ __device__ float2x3 make_float2x3(float2 c0, float2 c1, float2 c2)
{
	float2x3 t;

	t.c[0] = c0.x;
	t.c[1] = c0.y;
	t.c[2] = c1.x;
	t.c[3] = c1.y;
	t.c[4] = c2.x;
	t.c[5] = c2.y;
	return t;
}
inline __host__ __device__ float2x2 make_float2x2(float a, float b, float c, float d)
{
	float2x2 t;

	t.c[0] = a;
	t.c[1] = b;
	t.c[2] = c;
	t.c[3] = d;
	return t;
}
inline __host__ __device__ float2x2 make_float2x2(float2 c0, float2 c1)
{
	float2x2 t;

	t.c[0] = c0.x;
	t.c[1] = c0.y;
	t.c[2] = c1.x;
	t.c[3] = c1.y;
	return t;
}

inline __host__ __device__ float det(const float2x2& t)
{
	return t.c[0] * t.c[3] - t.c[1] * t.c[2];
}

inline __host__ __device__ float2x2 inverse(const float2x2& t)
{
	float detInv = 1.0f / det(t);

	return make_float2x2(
		make_float2(t.c[3], -t.c[1]) * detInv,
		make_float2(-t.c[2], t.c[0]) * detInv);
}

inline __host__ __device__ float2x2 operator+ (const float2x2& a, const float2x2& b)
{
	return make_float2x2(
		a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2], a.c[3] + b.c[3]
	);
}
inline __host__ __device__ float2x2 operator/ (const float2x2& a, float l)
{
	return make_float2x2(
		a.c[0] / l, a.c[1] / l, a.c[2] / l, a.c[3] / l
	);
}

inline __host__ __device__ float2x2 operator* (const float2x2& a, const float2x2& b)
{
	return make_float2x2(
		make_float2(
			a.c[0] * b.c[0] + a.c[2] * b.c[1],
			a.c[1] * b.c[0] + a.c[3] * b.c[1]),
		make_float2(
			a.c[0] * b.c[2] + a.c[2] * b.c[3],
			a.c[1] * b.c[2] + a.c[3] * b.c[3])
	);
}

inline __host__ __device__ float2x3 operator* (const float2x2& a, const float2x3& b)
{
	return make_float2x3(
		make_float2(
			a.c[0] * b.c[0] + a.c[2] * b.c[1],
			a.c[1] * b.c[0] + a.c[3] * b.c[1]),
		make_float2(
			a.c[0] * b.c[2] + a.c[2] * b.c[3],
			a.c[1] * b.c[2] + a.c[3] * b.c[3]),
		make_float2(
			a.c[0] * b.c[4] + a.c[2] * b.c[5],
			a.c[1] * b.c[4] + a.c[3] * b.c[5])
	);
}

inline __host__ __device__ float3x2 operator* (const float3x2& a, const float2x2& b)
{
	return make_float3x2(
		make_float3(
			a.c[0] * b.c[0] + a.c[3] * b.c[1],
			a.c[1] * b.c[0] + a.c[4] * b.c[1],
			a.c[2] * b.c[0] + a.c[5] * b.c[1]),
		make_float3(
			a.c[0] * b.c[2] + a.c[3] * b.c[3],
			a.c[1] * b.c[2] + a.c[4] * b.c[3],
			a.c[2] * b.c[2] + a.c[5] * b.c[3])
	);
}

inline __host__ __device__ float3 operator* (const float3x2& a, const float2& b)
{
	return make_float3(
		a.c[0] * b.x + a.c[3] * b.y,
		a.c[1] * b.x + a.c[4] * b.y,
		a.c[2] * b.x + a.c[5] * b.y);
}

/////////////////////////////////////////////////////////////////////////////////////////////
struct float3x3 {
	float c[9];

	inline __host__ __device__ float3 c0() const { return make_float3(c[0], c[1], c[2]); }
	inline __host__ __device__ float3 c1() const { return make_float3(c[3], c[4], c[5]); }
	inline __host__ __device__ float3 c2() const { return make_float3(c[6], c[7], c[8]); }

	float3x3& operator = (const float3x3& rhs)
	{
		for (int i = 0; i < 9; i++)
			c[i] = rhs.c[i];
		return *this;
	}
};

typedef struct __align__(16) {
	float3x3 u;
	float2 s;
	float2x2 vt;
} svd3x2;

inline __host__ __device__ float3 getCol(float3x3 a, int i)
{
	if (i == 0)
		return a.c0();
	else if (i == 1)
		return a.c1();
	else
		return a.c2();
}

inline __host__ __device__ float3x3 make_float3x3(float3 c0, float3 c1, float3 c2) {
	float3x3 t;
	t.c[0] = c0.x, t.c[1] = c0.y, t.c[2] = c0.z;
	t.c[3] = c1.x, t.c[4] = c1.y, t.c[5] = c1.z;
	t.c[6] = c2.x, t.c[7] = c2.y, t.c[8] = c2.z;
	return t;
}

inline __host__ __device__ void setIJ(float3x2& m, int i, int j, float v)
{
	m.c[j * 3 + i] = v;
}

inline __host__ __device__ float getIJ(const float3x3& m, int i, int j)
{
	return m.c[j * 3 + i];
}

inline __host__ __device__ float& getIJ(float3x3& m, int i, int j)
{
	return m.c[j * 3 + i];
}

inline __host__ __device__ float getIJ(const float2x2& m, int i, int j)
{
	return m.c[j * 2 + i];
}

inline __host__ __device__ float getI(const float2x2& m, int i)
{
	return m.c[i];
}

inline __host__ __device__ float& getI(float2x2& m, int i)
{
	return m.c[i];
}


inline __host__ void print_float3x3(float3x3* m) {
	//printf("%f ", m->c[0]); printf("%f ", m->c[3]); printf("%f \n", m->c[6]); 
	//printf("%f ", m->c[1]); printf("%f ", m->c[4]); printf("%f \n", m->c[7]); 
	//printf("%f ", m->c[2]); printf("%f ", m->c[5]); printf("%f \n\n", m->c[8]); 
}

inline __host__ __device__ float3x3 operator+ (const float3x3& m1, const float3x3& m2)
{
	return make_float3x3(
		m1.c0() + m2.c0(),
		m1.c1() + m2.c1(),
		m1.c2() + m2.c2());
}
inline __host__ __device__ void operator+= (float2x2& m1, const float2x2& m2)
{
	for (int i = 0; i < 4; i++)
		m1.c[i] += m2.c[i];
}
inline __host__ __device__ void operator+= (float3x3& m1, const float3x3& m2)
{
	for (int i = 0; i < 9; i++)
		m1.c[i] += m2.c[i];
}


inline __host__ __device__ float3x3 operator- (const float3x3& m1, const float3x3& m2) {
	return make_float3x3(
		m1.c0() - m2.c0(),
		m1.c1() - m2.c1(),
		m1.c2() - m2.c2());
}

inline __host__ __device__ float2x2 operator- (const float2x2& m1, const float2x2& m2) {
	return make_float2x2(
		m1.c0() - m2.c0(),
		m1.c1() - m2.c1());
}

inline __host__ __device__ float3x3 operator- (const float3x3& a) {
	return make_float3x3(-a.c0(), -a.c1(), -a.c2());
}


inline __host__ __device__ float3x3 operator* (float a, const float3x3& m) {
	return make_float3x3(a * m.c0(), a * m.c1(), a * m.c2());
}

inline __host__ __device__ float3x3 operator* (const float3x3& m, float a) {
	return make_float3x3(a * m.c0(), a * m.c1(), a * m.c2());
}

inline __host__ __device__ void operator*= (float3x3& m, float a) {
	for (int i = 0; i < 9; i++)
		m.c[i] *= a;
}

inline __host__ __device__ void operator*= (float2x2& m, float a) {
	for (int i = 0; i < 4; i++)
		m.c[i] *= a;
}
inline __host__ __device__ void operator/= (float2x2& m, float a) {
	for (int i = 0; i < 4; i++)
		m.c[i] /= a;
}
inline __host__ __device__ void operator-= (float2x2& m, const float2x2& n) {
	for (int i = 0; i < 4; i++)
		m.c[i] -= n.c[i];
}


inline __host__ __device__ float2x2 operator* (float a, const float2x2& m) {
	return make_float2x2(a * m.c0(), a * m.c1());
}

inline __host__ __device__ float2x2 operator* (const float2x2& m, float a) {
	return make_float2x2(a * m.c0(), a * m.c1());
}

inline __host__ __device__ float3x3 getTrans(const float3x3& m) {
	float3x3 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[3];
	t.c[2] = m.c[6];
	t.c[3] = m.c[1];
	t.c[4] = m.c[4];
	t.c[5] = m.c[7];
	t.c[6] = m.c[2];
	t.c[7] = m.c[5];
	t.c[8] = m.c[8];
	return t;
}

inline __host__ __device__ void getTrans(float t[], const float m[]) {
	t[0] = m[0];
	t[1] = m[3];
	t[2] = m[6];
	t[3] = m[1];
	t[4] = m[4];
	t[5] = m[7];
	t[6] = m[2];
	t[7] = m[5];
	t[8] = m[8];
}

inline __host__ __device__ float2x3 getTrans(const float3x2& m) {
	float2x3 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[3];
	t.c[2] = m.c[1];
	t.c[3] = m.c[4];
	t.c[4] = m.c[2];
	t.c[5] = m.c[5];
	return t;
}

inline __host__ __device__ float3x2 getTrans(const float2x3& m) {
	float3x2 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[2];
	t.c[2] = m.c[4];
	t.c[3] = m.c[1];
	t.c[4] = m.c[3];
	t.c[5] = m.c[5];
	return t;
}

inline __host__ __device__ float2x2 getTrans(const float2x2& m) {
	float2x2 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[2];
	t.c[2] = m.c[1];
	t.c[3] = m.c[3];
	return t;
}

inline __host__ __device__ float3x3 operator* (const float3x3& m1, const float3x3& m2) {
	float3x3 m1T = getTrans(m1);

	return make_float3x3(
		make_float3(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0()), dot(m1T.c2(), m2.c0())),
		make_float3(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1()), dot(m1T.c2(), m2.c1())),
		make_float3(dot(m1T.c0(), m2.c2()), dot(m1T.c1(), m2.c2()), dot(m1T.c2(), m2.c2())));
}

inline __host__ __device__ float3x2 operator* (const float3x3& m1, const float3x2& m2)
{
	float3x3 m1T = getTrans(m1);
	return make_float3x2(
		make_float3(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0()), dot(m1T.c2(), m2.c0())),
		make_float3(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1()), dot(m1T.c2(), m2.c1())));
}

inline __host__ __device__ float2x2 operator* (const float2x3& m1, const float3x2& m2) {
	float3x2 m1T = getTrans(m1);

	return make_float2x2(
		make_float2(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0())),
		make_float2(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1())));
}

inline __host__ __device__ float3x3 operator* (const float3x2& m1, const float2x3& m2) {
	float2x3 m1T = getTrans(m1);

	return make_float3x3(
		make_float3(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0()), dot(m1T.c2(), m2.c0())),
		make_float3(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1()), dot(m1T.c2(), m2.c1())),
		make_float3(dot(m1T.c0(), m2.c2()), dot(m1T.c1(), m2.c2()), dot(m1T.c2(), m2.c2())));
}

inline __host__ __device__ float3x3 outer(const float3& a, const float3& b) {
	return make_float3x3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3x3 identity3x3() {
	return make_float3x3(
		make_float3(1.0, 0.0, 0.0),
		make_float3(0.0, 1.0, 0.0),
		make_float3(0.0, 0.0, 1.0));
}

inline __host__ __device__ float2x2 identity2x2() {
	return make_float2x2(
		make_float2(1.0, 0.0),
		make_float2(0.0, 1.0));
}

inline __host__ __device__ float3x3 make_float3x3(float m) {
	return make_float3x3(
		make_float3(m, 0.0, 0.0),
		make_float3(0.0, m, 0.0),
		make_float3(0.0, 0.0, m)
	);
}

inline __host__ __device__ float2x2 make_float2x2(float m) {
	return make_float2x2(
		make_float2(m, 0.0),
		make_float2(0.0, m)
	);
}

inline __host__ __device__ float3x3 zero3x3() {
	return make_float3x3(
		make_float3(0.0, 0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(0.0, 0.0, 0.0)
	);
}

inline __host__ __device__ float2x2 zero2x2()
{
	return make_float2x2(
		make_float2(0, 0),
		make_float2(0, 0));
}

inline __host__ __device__ float3x2 zero3x2()
{
	return make_float3x2(
		make_float3(0, 0, 0),
		make_float3(0, 0, 0));
}

inline __host__ __device__ float3 getRow(float2x3 m, int i)
{
	if (i == 0)
		return make_float3(m.c[0], m.c[2], m.c[4]);
	if (i == 1)
		return make_float3(m.c[1], m.c[3], m.c[5]);
}

inline __host__ __device__ float3x3 getInverse(float3x3& m) {
	float3x3 t;
	t.c[0] = m.c[4] * m.c[8] - m.c[5] * m.c[7];
	t.c[1] = m.c[7] * m.c[2] - m.c[8] * m.c[1];
	t.c[2] = m.c[1] * m.c[5] - m.c[2] * m.c[4];

	t.c[3] = m.c[5] * m.c[6] - m.c[3] * m.c[8];
	t.c[4] = m.c[8] * m.c[0] - m.c[6] * m.c[2];
	t.c[5] = m.c[2] * m.c[3] - m.c[0] * m.c[5];

	t.c[6] = m.c[3] * m.c[7] - m.c[4] * m.c[6];
	t.c[7] = m.c[6] * m.c[1] - m.c[7] * m.c[0];
	t.c[8] = m.c[0] * m.c[4] - m.c[1] * m.c[3];

	float det = m.c[0] * t.c[0] + m.c[1] * t.c[3] + m.c[2] * t.c[6];
	float detInv = 1.0f / det;

	for (int i = 0; i < 9; i++)
		t.c[i] *= detInv;

	return t;
}

inline __host__ __device__ float3x3 rotation(float3& axis, float theta) {
	float s = sin(theta);
	float c = cos(theta);
	float t = 1 - c;
	float x = axis.x, y = axis.y, z = axis.z;

	float3x3 tt;
	tt.c[0] = t * x * x + c;
	tt.c[3] = t * x * y - s * z;
	tt.c[6] = t * x * z + s * y;
	tt.c[1] = t * x * y + s * z;
	tt.c[4] = t * y * y + c;
	tt.c[7] = t * y * z - s * x;
	tt.c[2] = t * x * z - s * y;
	tt.c[5] = t * y * z + s * x;
	tt.c[8] = t * z * z + c;

	return tt;
}

inline __host__ __device__ float3 operator* (const float3x3& m, const float3& rhs) {
	return make_float3(
		m.c[0] * rhs.x + m.c[3] * rhs.y + m.c[6] * rhs.z,
		m.c[1] * rhs.x + m.c[4] * rhs.y + m.c[7] * rhs.z,
		m.c[2] * rhs.x + m.c[5] * rhs.y + m.c[8] * rhs.z);
}

inline __host__ __device__ float2 operator* (const float2x2& m, const float2& rhs) {
	return make_float2(
		m.c[0] * rhs.x + m.c[2] * rhs.y,
		m.c[1] * rhs.x + m.c[3] * rhs.y);
}

inline __host__ __device__ int ad(const int3& a, int i) {
	if (i == 0) return a.x;
	if (i == 1) return a.y;
	if (i == 2) return a.z;
}

inline __host__ __device__ int& ad(int3& a, int i) {
	if (i == 0) return a.x;
	if (i == 1) return a.y;
	if (i == 2) return a.z;
}

inline __host__ __device__ float norm2(const float3& u) {
	return dot(u, u);
}

inline __host__ __device__ float norm2(const float2& u) {
	return dot(u, u);
}

inline __host__ __device__ float stp(const float3& u, const float3& v, const float3& w)
{
	return dot(u, cross(v, w));
}

#endif
