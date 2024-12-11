#include "utils.cuh"
#include "MainCodeDistanceQuery.cuh"
#include "TriDistance.cuh"
#include "TempData.cuh"
#include "Type.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

__host__ __device__ void getStartAndEnd(int tid, int deep, int numTri, int& start, int& end)
{
	start = 0;
	int length = numTri;
	for (int i = deep - 2; i >= 0; i--)
	{
		int lengthL = length - (length >> 1);
		if (tid & (1 << i))
		{
			start += lengthL;
			length >>= 1;
		}
		else {
			length = lengthL;
		}
	}
	end = start + length;
}


__host__ __device__ float
triDist(
	const float3& P1, const float3& P2, const float3& P3,
	const float3& Q1, const float3& Q2, const float3& Q3,
	float3& rP, float3& rQ);

__device__ void getMinMax(const g_box& a, const g_box& b, float& rmin);
__device__ void getMinMax_formax(const g_box& a, const g_box& b, float& rmin);
__device__ void getMinMax(const g_box& a, const g_box& b, float& rmin, float& rmax);
__global__ void kernel_gpu_init(g_box* bvhA, g_box* bvhB, float3* vtxsA, float3* vtxsB, g_bvtt_SoA buffer, int* g_length, int* g_length2, int id1, int id2, float* g_minDist, float dist)
{
	float newMin;
	getMinMax(bvhA[1], bvhB[1], newMin);
	float temp = norm2(vtxsA[id1 * 3] - vtxsB[id2 * 3]);
	g_minDist[0] = temp;
	buffer.a[0] = 1;
	buffer.b[0] = 1;
	buffer.min[0] = newMin;
	g_length[0] = 1;
	g_length2[0] = 0;
}

__global__ void kernel_gpu_init_formax(g_box* bvhA, g_box* bvhB, float3* vtxsA, float3* vtxsB, g_bvtt_SoA buffer, int* g_length, int* g_length2, float* g_minDist, float dist)
{
	float newMin;
	getMinMax_formax(bvhA[1], bvhB[1], newMin);
	float temp = norm2(vtxsA[0] - vtxsB[0]);
	g_minDist[0] = temp;
	buffer.a[0] = 1;
	buffer.b[0] = 1;
	buffer.min[0] = newMin;
	g_length[0] = 1;
	g_length2[0] = 0;
}

__global__ void kernel_updateVtx(float3* vtxs, float3* vtxsUpdate, const g_transf trans, int num)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num) return;
	vtxsUpdate[tid] = trans.apply(vtxs[tid]);
}

__global__ void kernel_updateBV_ForMax(g_box* nodes, float3* vtxs, int* cnt, const int deep, const int numTri)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= (1 << (deep - 1))) return;
	int nodeId = tid + (1 << (deep - 1));
	int leafId = tid;

	int2 leaf;
	getStartAndEnd(leafId, deep, numTri, leaf.x, leaf.y);
	float3 v0, v1, v2;
	g_box box;
	v0 = vtxs[leaf.x];
	box.set(v0);
	if (leaf.x + 1 < leaf.y)
	{
		v0 = vtxs[leaf.x + 1];
		box.add(v0);
	}

	nodes[nodeId] = box;

	int ret;
	while (nodeId > 1)
	{
		__threadfence();
		int j = (nodeId >> 1);
		ret = atomicAdd(cnt + j, 1);
		if (ret % 2 == 0) {
			break;
		}
		else {
			box.add(nodes[nodeId ^ 1]);
		}
		nodes[j] = box;
		nodeId = j;
	}
}

__global__ void kernel_updateBV(g_box* nodes, float3* vtxs, int* cnt, const int deep, const int numTri)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= (1 << (deep - 1))) return;
	int nodeId = tid + (1 << (deep - 1));
	int leafId = tid;

	int2 leaf;
	getStartAndEnd(leafId, deep, numTri, leaf.x, leaf.y);
	float3 v0, v1, v2;
	g_box box;
	v0 = vtxs[leaf.x * 3];
	v1 = vtxs[leaf.x * 3 + 1];
	v2 = vtxs[leaf.x * 3 + 2];
	box.set(v0);
	box.add(v1);
	box.add(v2);
	if (leaf.x + 1 < leaf.y)
	{
		v0 = vtxs[leaf.x * 3 + 3];
		v1 = vtxs[leaf.x * 3 + 4];
		v2 = vtxs[leaf.x * 3 + 5];
		box.add(v0);
		box.add(v1);
		box.add(v2);

	}

	nodes[nodeId] = box;

	int ret;
	while (nodeId > 1)
	{
		__threadfence();
		int j = (nodeId >> 1);
		ret = atomicAdd(cnt + j, 1);
		if (ret % 2 == 0) {
			break;
		}
		else {
			box.add(nodes[nodeId ^ 1]);
		}
		nodes[j] = box;
		nodeId = j;
	}
}

__device__ float getSquareDistance(float a, float b, float c, float d)
{
	return (a - c) * (a - c) + (b - d) * (b - d);
}

__device__ float getSquareDistance(float x1, float y1, float z1, float x2, float y2, float z2)
{
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

__device__ float getMaxParallel(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l)
{
	float ret = getSquareDistance(a, c, f, h);
	ret = fmax(ret, getSquareDistance(a, d, f, g));
	ret = fmax(ret, getSquareDistance(b, c, e, h));
	ret = fmax(ret, getSquareDistance(b, d, e, g));
	float dis = abs(k - i);
	dis = min(dis, abs(i - l));
	dis = min(dis, abs(j - k));
	dis = min(dis, abs(j - l));
	return ret + dis * dis;
}

__device__ float getMaxPlaneXY(float x1,               float mny1, float mxy1, float mnz1, float mxz1,
	                           float mnx2, float mxx2, float y2,               float mnz2, float mxz2)
{
	float ret = getSquareDistance(x1, mny1, mnz1, mnx2, y2, mxz2);
	ret = fmax(ret, getSquareDistance(x1, mny1, mnz1, mxx2, y2, mxz2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mnz1, mnx2, y2, mxz2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mnz1, mxx2, y2, mxz2));
	ret = fmax(ret, getSquareDistance(x1, mny1, mxz1, mnx2, y2, mnz2));
	ret = fmax(ret, getSquareDistance(x1, mny1, mxz1, mxx2, y2, mnz2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mxz1, mnx2, y2, mnz2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mxz1, mxx2, y2, mnz2));
	return ret;
}

__device__ float getMaxPlaneXZ(float x1,               float mny1, float mxy1, float mnz1, float mxz1,
	                           float mnx2, float mxx2, float mny2, float mxy2, float z2)
{
	float ret = getSquareDistance(x1, mny1, mnz1, mnx2, mxy2, z2);
	ret = fmax(ret, getSquareDistance(x1, mny1, mnz1, mxx2, mxy2, z2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mnz1, mnx2, mny2, z2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mnz1, mxx2, mny2, z2));
	ret = fmax(ret, getSquareDistance(x1, mny1, mxz1, mnx2, mxy2, z2));
	ret = fmax(ret, getSquareDistance(x1, mny1, mxz1, mxx2, mxy2, z2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mxz1, mnx2, mny2, z2));
	ret = fmax(ret, getSquareDistance(x1, mxy1, mxz1, mxx2, mny2, z2));
	return ret;
}

__device__ float getMaxPlaneYZ(float mnx1, float mxx1, float y1, float mnz1, float mxz1,
	                           float mnx2, float mxx2, float mny2, float mxy2, float z2)
{
	float ret = getSquareDistance(mxx1, y1, mnz1, mnx2, mxy2, z2);
	ret = fmax(ret, getSquareDistance(mnx1, y1, mnz1, mxx2, mxy2, z2));
	ret = fmax(ret, getSquareDistance(mxx1, y1, mnz1, mnx2, mny2, z2));
	ret = fmax(ret, getSquareDistance(mnx1, y1, mnz1, mxx2, mny2, z2));
	ret = fmax(ret, getSquareDistance(mxx1, y1, mxz1, mnx2, mxy2, z2));
	ret = fmax(ret, getSquareDistance(mnx1, y1, mxz1, mxx2, mxy2, z2));
	ret = fmax(ret, getSquareDistance(mxx1, y1, mxz1, mnx2, mny2, z2));
	ret = fmax(ret, getSquareDistance(mnx1, y1, mxz1, mxx2, mny2, z2));
	return ret;
}

__device__ void getMinMax(const g_box& a, const g_box& b, float& rmin)
{
	float lx, ly, lz;
	if (a._max.x < b._min.x)
	{
		lx = b._min.x - a._max.x;
	}
	else if (b._max.x < a._min.x)
	{
		lx = a._min.x - b._max.x;
	}
	else
	{
		lx = 0;
	}

	if (a._max.y < b._min.y)
	{
		ly = b._min.y - a._max.y;
	}
	else if (b._max.y < a._min.y)
	{
		ly = a._min.y - b._max.y;
	}
	else
	{
		ly = 0;
	}

	if (a._max.z < b._min.z)
	{
		lz = b._min.z - a._max.z;
	}
	else if (b._max.z < a._min.z)
	{
		lz = a._min.z - b._max.z;
	}
	else
	{
		lz = 0;
	}

	rmin = lx * lx + ly * ly + lz * lz;

#ifdef USE_14Dop

	float temp = 0;
	temp = max(temp, a.minCorner.x - b.maxCorner.x);
	temp = max(temp, b.minCorner.x - a.maxCorner.x);
	temp = max(temp, a.minCorner.y - b.maxCorner.y);
	temp = max(temp, b.minCorner.y - a.maxCorner.y);
	temp = max(temp, a.minCorner.z - b.maxCorner.z);
	temp = max(temp, b.minCorner.z - a.maxCorner.z);
	temp = max(temp, a.minCorner.w - b.maxCorner.w);
	temp = max(temp, b.minCorner.w - a.maxCorner.w);
	
	rmin = max(temp * temp / 3, rmin);

#endif // DEBUG
}

__device__ void getMinMax_formax(const g_box& a, const g_box& b, float& ret)
{
	float lx, ly, lz;
	float3 t1 = a._max - b._min;
	float3 t2 = b._max - a._min;
	ret = norm2(fmaxf(t1, t2));
}

__device__ void getMinMax(const g_box& a, const g_box& b, float& rmin, float& rmax)
{
	float lx, ly, lz;
	if (a._max.x < b._min.x)
	{
		lx = b._min.x - a._max.x;
	}
	else if (b._max.x < a._min.x)
	{
		lx = a._min.x - b._max.x;
	}
	else
	{
		lx = 0;
	}

	if (a._max.y < b._min.y)
	{
		ly = b._min.y - a._max.y;
	}
	else if (b._max.y < a._min.y)
	{
		ly = a._min.y - b._max.y;
	}
	else
	{
		ly = 0;
	}

	if (a._max.z < b._min.z)
	{
		lz = b._min.z - a._max.z;
	}
	else if (b._max.z < a._min.z)
	{
		lz = a._min.z - b._max.z;
	}
	else
	{
		lz = 0;
	}

	rmin = lx * lx + ly * ly + lz * lz;
	rmax = getMaxParallel(a._min.x, a._max.x, a._min.y, a._max.y, b._min.x, b._max.x, b._min.y, b._max.y, a._min.z, a._max.z, b._min.z, b._max.z);
	rmax = fmin(rmax, getMaxParallel(a._min.y, a._max.y, a._min.z, a._max.z, b._min.y, b._max.y, b._min.z, b._max.z, a._min.x, a._max.x, b._min.x, b._max.x));
	rmax = fmin(rmax, getMaxParallel(a._min.x, a._max.x, a._min.z, a._max.z, b._min.x, b._max.x, b._min.z, b._max.z, a._min.y, a._max.y, b._min.y, b._max.y));
	rmax = fmin(rmax, getMaxPlaneXY(a._min.x,           a._min.y, a._max.y, a._min.z, a._max.z,
		                            b._min.x, b._max.x, b._min.y,           b._min.z, b._max.z));
	rmax = fmin(rmax, getMaxPlaneXY(a._min.x,           a._min.y, a._max.y, a._min.z, a._max.z,
		                            b._min.x, b._max.x, b._max.y,           b._min.z, b._max.z));
	rmax = fmin(rmax, getMaxPlaneXY(a._max.x, a._min.y, a._max.y, a._min.z, a._max.z,
		                            b._min.x, b._max.x, b._min.y, b._min.z, b._max.z));
	rmax = fmin(rmax, getMaxPlaneXY(a._max.x, a._min.y, a._max.y, a._min.z, a._max.z,
		                            b._min.x, b._max.x, b._max.y, b._min.z, b._max.z));
	
	rmax = fmin(rmax, getMaxPlaneXY(b._min.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._min.z, a._max.z));
	rmax = fmin(rmax, getMaxPlaneXY(b._min.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._max.y, a._min.z, a._max.z));
	rmax = fmin(rmax, getMaxPlaneXY(b._max.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._min.z, a._max.z));
	rmax = fmin(rmax, getMaxPlaneXY(b._max.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._max.y, a._min.z, a._max.z));
	
	rmax = fmin(rmax, getMaxPlaneYZ(a._min.x, a._max.x, a._min.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._min.z));
	rmax = fmin(rmax, getMaxPlaneYZ(a._min.x, a._max.x, a._min.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._max.z));
	rmax = fmin(rmax, getMaxPlaneYZ(a._min.x, a._max.x, a._max.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._min.z));
	rmax = fmin(rmax, getMaxPlaneYZ(a._min.x, a._max.x, a._max.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._max.z));
	
	rmax = fmin(rmax, getMaxPlaneYZ(b._min.x, b._max.x, b._min.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._min.z));
	rmax = fmin(rmax, getMaxPlaneYZ(b._min.x, b._max.x, b._min.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._max.z));
	rmax = fmin(rmax, getMaxPlaneYZ(b._min.x, b._max.x, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._min.z));
	rmax = fmin(rmax, getMaxPlaneYZ(b._min.x, b._max.x, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._max.z));
	
	rmax = fmin(rmax, getMaxPlaneXZ(a._min.x, a._min.y, a._max.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._min.z));
	rmax = fmin(rmax, getMaxPlaneXZ(a._min.x, a._min.y, a._max.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._max.z));
	rmax = fmin(rmax, getMaxPlaneXZ(a._max.x, a._min.y, a._max.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._min.z));
	rmax = fmin(rmax, getMaxPlaneXZ(a._max.x, a._min.y, a._max.y, a._min.z, a._max.z,
		b._min.x, b._max.x, b._min.y, b._max.y, b._max.z));
	
	rmax = fmin(rmax, getMaxPlaneXZ(b._min.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._min.z));
	rmax = fmin(rmax, getMaxPlaneXZ(b._min.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._max.z));
	rmax = fmin(rmax, getMaxPlaneXZ(b._max.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._min.z));
	rmax = fmin(rmax, getMaxPlaneXZ(b._max.x, b._min.y, b._max.y, b._min.z, b._max.z,
		a._min.x, a._max.x, a._min.y, a._max.y, a._max.z));
	//printf("ret %f\n", rmax);
}

inline __device__ float atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
 
inline __device__ float atomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__host__ __device__ int calProDeep(int maxDeepA, int maxDeepB, int bvttLength, int deepNow)
{
	int ret = 1, maxDeep;
	if (maxDeepA > maxDeepB)
	{
		int t = maxDeepA; maxDeepA = maxDeepB; maxDeepB = t;
	}
	if (deepNow < maxDeepA && deepNow < maxDeepB)
	{
		maxDeep = min(maxDeepA - deepNow, maxDeepB - deepNow);
		int temp = 1024 * 128 * 2 / bvttLength;
		while ((1 << ret) * (1 << ret) < temp && deepNow + ret < maxDeepA && deepNow + ret < maxDeepB) ret++;
	}
	else
	{
		maxDeep = max(maxDeepA - deepNow, maxDeepB - deepNow);
		int temp = 1024 * 128 * 2 / bvttLength;
		while ((1 << ret) < temp && deepNow + ret < maxDeepB) ret++;
	}
	//if (ret > 3) ret = ret * 0.7;
	//if (ret != 9) ret = 3;
	//maxDeep = min(maxDeep, 7);
	return min(ret, maxDeep);
}

__global__ void kernel_fast_traversal_formax(g_box* bvhA, g_box* bvhB, float3* vtxsA, float3* vtxsB, int numTriA, int numTriB, int deepNow, g_bvtt_SoA bvtt_in, g_bvtt_SoA bvtt_out, int* g_length, float* g_minDist, int bvtt_in_length, int deep)
{
	__shared__ int start;
	//__shared__ int scan[128];
	typedef cub::BlockReduce<float, 128> BlockReduce;
	typedef cub::BlockScan<int, 128> BlockScan;
	__shared__ typename BlockReduce::TempStorage reduce_temp_storage;
	__shared__ typename BlockScan::TempStorage scan_temp_storage;


	const int group_length = (1 << (deep << 1));
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	const int group_id = tid / group_length;
	const int group_rank = tid % group_length;

	int a; //bvtt.bvtt.x;
	int b; //bvtt.bvtt.y;
	float min;
	int cnt = 0;
	//g_bvtt bnode;
	int newA, newB;
	float my_minDist = g_minDist[0];
	float newMin, newMax = my_minDist;
	if (tid < group_length * bvtt_in_length) {
		a = bvtt_in.a[group_id];
		b = bvtt_in.b[group_id];
		min = bvtt_in.min[group_id];
	}

	if (min < my_minDist || tid >= group_length * bvtt_in_length)
	{
	}
	else
	{
		int sonsA = (a << deep);
		int sonsB = (b << deep);
		int startA, endA, startB, endB;
		newA = (group_rank >> deep) + sonsA;
		newB = (group_rank % (1 << deep)) + sonsB;
		getMinMax_formax(bvhA[newA], bvhB[newB], newMin);

		getStartAndEnd(newA - (1 << (deepNow + deep - 1)), deepNow + deep, numTriA, startA, endA);
		getStartAndEnd(newB - (1 << (deepNow + deep - 1)), deepNow + deep, numTriB, startB, endB);

		float3 vAs0 = vtxsA[startA];
		float3 vBs0 = vtxsB[startB];
		newMax = norm2(vAs0 - vBs0);
		if (newMin >= my_minDist)
		{
			cnt = 1;
		}
	}

	//scan[threadIdx.x] = cnt;

	float aggregate = BlockReduce(reduce_temp_storage).Reduce(newMax, cub::Max());

	int temp;
	BlockScan(scan_temp_storage).InclusiveSum(cnt, temp);

	if (threadIdx.x == 127)
	{
		start = atomicAdd(g_length, temp);
	}
	__syncthreads();
	if (cnt != 0) {
		//int bias = start + scan[threadIdx.x] - 1;
		int bias = start + temp - 1;
		bvtt_out.a[bias] = newA;
		bvtt_out.b[bias] = newB;
		bvtt_out.min[bias] = newMin;
	}
	if (threadIdx.x == 0)
	{
		atomicMax(g_minDist, aggregate);
	}
}

__global__ void kernel_fast_traversal(g_box* bvhA, g_box* bvhB, float3* vtxsA, float3* vtxsB, int numTriA, int numTriB, int deepNow, g_bvtt_SoA bvtt_in, g_bvtt_SoA bvtt_out, int* g_length, float* g_minDist, int bvtt_in_length, int deep)
{
	__shared__ int start;
	//__shared__ int scan[128];
	typedef cub::BlockReduce<float, 128> BlockReduce;
	typedef cub::BlockScan<int, 128> BlockScan;
	__shared__ typename BlockReduce::TempStorage reduce_temp_storage;
	__shared__ typename BlockScan::TempStorage scan_temp_storage;

	
	const int group_length = (1 << (deep << 1));
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	const int group_id = tid / group_length;
	const int group_rank = tid % group_length;

	int a; //bvtt.bvtt.x;
	int b; //bvtt.bvtt.y;
	float min;
	int cnt = 0;
	//g_bvtt bnode;
	int newA, newB;
	float my_minDist = g_minDist[0];
	float newMin, newMax = my_minDist;
	if (tid < group_length * bvtt_in_length) {
		a = bvtt_in.a[group_id];
		b = bvtt_in.b[group_id];
		min = bvtt_in.min[group_id];
	}

	if (min > my_minDist || tid >= group_length * bvtt_in_length)
	{
	}
	else
	{
		int sonsA = (a << deep);
		int sonsB = (b << deep);
		int startA, endA, startB, endB;
		newA = (group_rank >> deep) + sonsA;
		newB = (group_rank % (1 << deep)) + sonsB;
		getMinMax(bvhA[newA], bvhB[newB], newMin);
		
		getStartAndEnd(newA - (1 << (deepNow + deep - 1)), deepNow + deep, numTriA, startA, endA);
		getStartAndEnd(newB - (1 << (deepNow + deep - 1)), deepNow + deep, numTriB, startB, endB);

		float3 vAs0 = vtxsA[startA * 3];
		float3 vBs0 = vtxsB[startB * 3];
		////float3 vAs1 = vtxsA[startA * 3 + 1];
		//float3 vBs1 = vtxsB[startB * 3 + 1];
		//float3 vAs2 = vtxsA[startA * 3 + 2];
		//float3 vBs2 = vtxsB[startB * 3 + 2];
		newMax = norm2(vAs0 - vBs0);
		//newMax = fmin(newMax, norm2(vAs0 - vBs1));
		//newMax = fmin(newMax, norm2(vAs0 - vBs2));
		//newMax = fmin(newMax, norm2(vAs1 - vBs0));
		//newMax = fmin(newMax, norm2(vAs1 - vBs1));
		//newMax = fmin(newMax, norm2(vAs1 - vBs2));
		//newMax = fmin(newMax, norm2(vAs2 - vBs0));
		//newMax = fmin(newMax, norm2(vAs2 - vBs1));
		//newMax = fmin(newMax, norm2(vAs2 - vBs2));
		//newMax = fmin(newMax, norm2(vAe - vBs));
		//newMax = fmin(newMax, norm2(vAe - vBe));
		if (newMin <= my_minDist)
		{
			cnt = 1;
		}
	}

	//scan[threadIdx.x] = cnt;

	float aggregate = BlockReduce(reduce_temp_storage).Reduce(newMax, cub::Min());

	int temp;
    BlockScan(scan_temp_storage).InclusiveSum(cnt, temp);
	//__syncthreads();
	//int temp = scan[threadIdx.x];
	//for (int stride = 1; stride < 128; stride *= 2) {
	//	__syncthreads();
	//	if (threadIdx.x >= stride)
	//		temp = scan[threadIdx.x] + scan[threadIdx.x - stride];
	//	__syncthreads();
	//	scan[threadIdx.x] = temp; 
	//}
	if (threadIdx.x == 127)
	{
		start = atomicAdd(g_length, temp);
	}
	__syncthreads();
	if (cnt != 0) {
		//int bias = start + scan[threadIdx.x] - 1;
		int bias = start + temp - 1;
		bvtt_out.a[bias] = newA;
		bvtt_out.b[bias] = newB;
		bvtt_out.min[bias] = newMin;
	}
	if (threadIdx.x == 0)
	{
		atomicMin(g_minDist, aggregate);
	}
}

__global__ void kernel_fast_traversal_onlyB(g_box* bvhA, g_box* bvhB, float3* vtxsA, float3* vtxsB, int numTriA, int numTriB, int deepA, int deepNow, g_bvtt_SoA bvtt_in, g_bvtt_SoA bvtt_out, int* g_length, float* g_minDist, int bvtt_in_length, int deep)
{
	__shared__ int start;
	typedef cub::BlockReduce<float, 128> BlockReduce;
	typedef cub::BlockScan<int, 128> BlockScan;
	__shared__ typename BlockReduce::TempStorage reduce_temp_storage;
	__shared__ typename BlockScan::TempStorage scan_temp_storage;

	const int group_length = (1 << deep);
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	const int group_id = tid / group_length;
	const int group_rank = tid % group_length;

	int a; //bvtt.bvtt.x;
	int b; //bvtt.bvtt.y;
	float min;
	int cnt = 0;

	int newA, newB;
	float my_minDist = g_minDist[0];
	float newMin, newMax = my_minDist;
	if (tid < group_length * bvtt_in_length) {
		a = bvtt_in.a[group_id];
		b = bvtt_in.b[group_id];
		min = bvtt_in.min[group_id];
	}

	if (min > my_minDist || tid >= group_length * bvtt_in_length)
	{
	}
	else
	{
		int sonsB = (b << deep);
		newA = a;
		newB = group_rank + sonsB;
		int startA, endA, startB, endB;
		getMinMax(bvhA[newA], bvhB[newB], newMin);

		getStartAndEnd(newA - (1 << (deepA - 1)), deepA, numTriA, startA, endA);
		getStartAndEnd(newB - (1 << (deepNow + deep - 1)), deepNow + deep, numTriB, startB, endB);
		float3 vAs = vtxsA[startA * 3];
		float3 vBs = vtxsB[startB * 3];
		newMax = norm2(vAs - vBs);
		if (newMin <= my_minDist)
		{
			cnt = 1;
		}
	}


	float aggregate = BlockReduce(reduce_temp_storage).Reduce(newMax, cub::Min());

	int temp;
	BlockScan(scan_temp_storage).InclusiveSum(cnt, temp);

	if (threadIdx.x == 127)
	{
		start = atomicAdd(g_length, temp);
	}
	__syncthreads();
	if (cnt != 0) {
		int bias = start + temp - 1;
		bvtt_out.a[bias] = newA;
		bvtt_out.b[bias] = newB;
		bvtt_out.min[bias] = newMin;
	}
	if (threadIdx.x == 0)
	{
		atomicMin(g_minDist, aggregate);
	}
}

__global__ void kernel_fast_traversal_v2(g_box* bvhA, g_box* bvhB, float3* vtxsA, float3* vtxsB, int numTriA, int numTriB, int deepNow, g_bvtt_SoA bvtt_in, g_bvtt_SoA bvtt_out, int* g_length_in, int* g_length_out, float* g_minDist)
{
	__shared__ int start;
	//__shared__ int scan[128];
	typedef cub::BlockReduce<float, 128> BlockReduce;
	typedef cub::BlockScan<int, 128> BlockScan;
	__shared__ typename BlockReduce::TempStorage reduce_temp_storage;
	__shared__ typename BlockScan::TempStorage scan_temp_storage;

	int bvtt_in_length = g_length_in[0];
	int deep = calProDeep(20, 20, bvtt_in_length, deepNow);

	const int group_length = (1 << (deep << 1));
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	const int group_id = tid / group_length;
	const int group_rank = tid % group_length;

	int a; //bvtt.bvtt.x;
	int b; //bvtt.bvtt.y;
	float min;
	int cnt = 0;
	//g_bvtt bnode;
	int newA, newB;
	float my_minDist = g_minDist[0];
	float newMin, newMax = my_minDist;
	if (tid < group_length * bvtt_in_length) {
		a = bvtt_in.a[group_id];
		b = bvtt_in.b[group_id];
		min = bvtt_in.min[group_id];
	}

	if (min > my_minDist || tid >= group_length * bvtt_in_length)
	{
	}
	else
	{
		int sonsA = (a << deep);
		int sonsB = (b << deep);
		int startA, endA, startB, endB;
		newA = (group_rank >> deep) + sonsA;
		newB = (group_rank % (1 << deep)) + sonsB;
		getMinMax(bvhA[newA], bvhB[newB], newMin);

		getStartAndEnd(newA - (1 << (deepNow + deep - 1)), deepNow + deep, numTriA, startA, endA);
		getStartAndEnd(newB - (1 << (deepNow + deep - 1)), deepNow + deep, numTriB, startB, endB);

		float3 vAs = vtxsA[startA * 3];
		float3 vBs = vtxsB[startB * 3];
		//loat3 vAe = vtxsA[endA * 3 - 3];
		//loat3 vBe = vtxsB[endB * 3 - 3];
		newMax = norm2(vAs - vBs);
		//newMax = fmin(newMax, norm2(vAs - vBe));
		//newMax = fmin(newMax, norm2(vAe - vBs));
		//newMax = fmin(newMax, norm2(vAe - vBe));
		if (newMin <= my_minDist)
		{
			cnt = 1;
		}
	}

	//scan[threadIdx.x] = cnt;

	float aggregate = BlockReduce(reduce_temp_storage).Reduce(newMax, cub::Min());

	int temp;
	BlockScan(scan_temp_storage).InclusiveSum(cnt, temp);
	//__syncthreads();
	//int temp = scan[threadIdx.x];
	//for (int stride = 1; stride < 128; stride *= 2) {
	//	__syncthreads();
	//	if (threadIdx.x >= stride)
	//		temp = scan[threadIdx.x] + scan[threadIdx.x - stride];
	//	__syncthreads();
	//	scan[threadIdx.x] = temp; 
	//}
	if (threadIdx.x == 127)
	{
		start = atomicAdd(g_length_out, temp);
	}
	__syncthreads();
	if (cnt != 0) {
		//int bias = start + scan[threadIdx.x] - 1;
		int bias = start + temp - 1;
		bvtt_out.a[bias] = newA;
		bvtt_out.b[bias] = newB;
		bvtt_out.min[bias] = newMin;
	}
	if (threadIdx.x == 0)
	{
		atomicMin(g_minDist, aggregate);
	}
}

__host__ __device__ float
TriDistPQP(const float3* S, const float3* T);

__device__ float raw_distance_simple(const float3* S, const float3* T)
{
	float ret;
	int point;
	{
		float3 N = cross(T[1] - T[0], T[2] - T[0]);
		float Nl = norm2(N);

		float t0, t1, t2;
		t0 = dot(N, S[0] - T[0]);
		t1 = dot(N, S[1] - T[0]);
		t2 = dot(N, S[2] - T[0]);

		float temp = 0;
		if (t0 > 0 && t1 > 0 && t2 > 0)
		{
			temp = t0;
			if (t1 > temp) temp = t1;
			if (t2 > temp) temp = t2;
		}
		else if (t0 < 0 && t1 < 0 && t2 < 0)
		{
			temp = t0;
			if (t1 > temp) temp = t1;
			if (t2 > temp) temp = t2;
		}

		ret = temp * temp / Nl;

		N = cross(S[1] - S[0], S[2] - S[0]);
		Nl = norm2(N);
		t0 = dot(N, T[0] - S[0]);
		t1 = dot(N, T[1] - S[0]);
		t2 = dot(N, T[2] - S[0]);

		if (t0 > 0 && t1 > 0 && t2 > 0)
		{
			temp = t0;
			if (t1 < temp) temp = t1;
			if (t2 < temp) temp = t2;
		}
		else if (t0 < 0 && t1 < 0 && t2 < 0)
		{
			temp = t0;
			if (t1 > temp) temp = t1;
			if (t2 > temp) temp = t2;
		}

		if (temp * temp / Nl > ret) ret = temp * temp / Nl;
	}
	return ret;
}

__global__ void kernel_narrow_test_formax(g_box* bvhA, g_box* bvhB, float3* vtxA, float3* vtxB, g_bvtt_SoA bvttNode, float* out_dis, float* g_minDist, int deepA, int deepB, int numTriA, int numTriB, int num)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num) return;

	//g_bvtt bvtt = bvttNode[tid];
	int a, b;
	a = bvttNode.a[tid / 4];
	b = bvttNode.b[tid / 4];

	float newMin;
	getMinMax_formax(bvhA[a], bvhB[b], newMin);

	if (newMin < g_minDist[0]) {
		out_dis[tid] = -2; return;
	}

	int2 a_leaf;
	getStartAndEnd(a - (1 << (deepA - 1)), deepA, numTriA, a_leaf.x, a_leaf.y);
	int2 b_leaf;
	getStartAndEnd(b - (1 << (deepB - 1)), deepB, numTriB, b_leaf.x, b_leaf.y);

	float3 v0, v1, v2;
	float3 v3, v4, v5;
	float mmin = g_minDist[0];
	float3 p, q;

	int as = tid & 1;
	int bs = (tid & 2) >> 1;

	if (a_leaf.x + as >= a_leaf.y || b_leaf.x + bs >= b_leaf.y) {
		out_dis[tid] = -1; return;
	}

	a_leaf.x += as;
	b_leaf.x += bs;

	out_dis[tid] = norm2(vtxA[a_leaf.x] - vtxB[b_leaf.x ]);
}

__global__ void kernel_narrow_test(g_box* bvhA, g_box* bvhB, float3* vtxA, float3* vtxB, g_bvtt_SoA bvttNode, float* out_dis, int* out_id1, int* out_id2, float* g_minDist, int deepA, int deepB, int numTriA, int numTriB, int num)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num) return;

	//g_bvtt bvtt = bvttNode[tid];
	int a, b;
	a = bvttNode.a[tid / 4];
	b = bvttNode.b[tid / 4];
	
	float newMin;
	getMinMax(bvhA[a], bvhB[b], newMin);
	
	int2 a_leaf;
	getStartAndEnd(a - (1 << (deepA - 1)), deepA, numTriA, a_leaf.x, a_leaf.y);
	int2 b_leaf;
	getStartAndEnd(b - (1 << (deepB - 1)), deepB, numTriB, b_leaf.x, b_leaf.y);

	if (newMin > g_minDist[0]) {
		out_dis[tid] = 10000000; return;
	}

	

	float3 v0, v1, v2;
	float3 v3, v4, v5;
	float mmin = g_minDist[0];
	float3 p, q;

	int as = tid & 1;
	int bs = (tid & 2) >> 1;

	if (a_leaf.x + as >= a_leaf.y || b_leaf.x + bs >= b_leaf.y) {
		out_dis[tid] = 10000000; return;
	}

	a_leaf.x += as;
	b_leaf.x += bs;

	float3 S[3];
	float3 T[3];

	S[0] = vtxA[a_leaf.x * 3];
	S[1] = vtxA[a_leaf.x * 3 + 1];
	S[2] = vtxA[a_leaf.x * 3 + 2];

	T[0] = vtxB[b_leaf.x * 3];
	T[1] = vtxB[b_leaf.x * 3 + 1];
	T[2] = vtxB[b_leaf.x * 3 + 2];

	float3 rP, rQ;

	//float temp = raw_distance_simple(S, T);
	//if (temp < mmin)
	//{
	mmin = TriDistPQP(S, T);
	//}
	out_dis[tid] = mmin;
	
}

__global__ void kernel_findId(g_box* bvhA, g_box* bvhB, float3* vtxA, float3* vtxB, g_bvtt_SoA bvttNode, float* out_dis, int* out_id1, int* out_id2, float* g_minDist, int deepA, int deepB, int numTriA, int numTriB, int num)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num) return;

	//g_bvtt bvtt = bvttNode[tid];
	int a, b;
	a = bvttNode.a[tid / 4];
	b = bvttNode.b[tid / 4];

	int2 a_leaf;
	getStartAndEnd(a - (1 << (deepA - 1)), deepA, numTriA, a_leaf.x, a_leaf.y);
	int2 b_leaf;
	getStartAndEnd(b - (1 << (deepB - 1)), deepB, numTriB, b_leaf.x, b_leaf.y);



	float3 v0, v1, v2;
	float3 v3, v4, v5;
	float mmin = g_minDist[0];
	float3 p, q;

	int as = tid & 1;
	int bs = (tid & 2) >> 1;

	if (a_leaf.x + as >= a_leaf.y || b_leaf.x + bs >= b_leaf.y) {
		return;
	}

	a_leaf.x += as;
	b_leaf.x += bs;

	float3 S[3];
	float3 T[3];

	S[0] = vtxA[a_leaf.x * 3];
	S[1] = vtxA[a_leaf.x * 3 + 1];
	S[2] = vtxA[a_leaf.x * 3 + 2];

	T[0] = vtxB[b_leaf.x * 3];
	T[1] = vtxB[b_leaf.x * 3 + 1];
	T[2] = vtxB[b_leaf.x * 3 + 2];

	float3 rP, rQ;

	//float temp = raw_distance_simple(S, T);
	//if (temp < mmin)
	//{
	mmin = TriDistPQP(S, T);
	if (mmin == g_minDist[0])
	{
		out_id1[0] = a_leaf.x;
		out_id2[0] = b_leaf.x;
	}

}

__inline__ __host__ __device__ void
SegPointsGPU(float3& X, float3& Y, // closest points
	const float3& P, const float3& A, // seg 1 origin, vector
	const float3& Q, const float3& B);

__device__ float raw_distance(const float3* S, const float3* T, bool pr = false)
{
	float3 Sn = cross(S[1] - S[0], S[2] - S[0]);
	float3 Tn = cross(T[1] - T[0], T[2] - T[0]);

	float3 Zn = cross(Sn, Tn);
	float Znl = norm2(Zn);
	if (Znl < 1e-6)
	{
		return 0;
		Zn = make_float3(1, 0, 0);
		Znl = 1;
	}

	float ret = 0;
	{
		float SZmx = dot(S[0], Zn);
		float SZmn = SZmx;
		float temp = dot(S[1], Zn);
		if (temp > SZmx) SZmx = temp; else if (temp < SZmn) SZmn = temp;
		temp = dot(S[2], Zn);
		if (temp > SZmx) SZmx = temp; else if (temp < SZmn) SZmn = temp;

		float TZmx = dot(T[0], Zn);
		float TZmn = TZmx;
		temp = dot(T[1], Zn);
		if (temp > TZmx) TZmx = temp; else if (temp < TZmn) TZmn = temp;
		temp = dot(T[2], Zn);
		if (temp > TZmx) TZmx = temp; else if (temp < TZmn) TZmn = temp;

		ret = max(ret, TZmn - SZmx);
		ret = max(ret, SZmn - TZmx);

		ret = ret * ret / Znl;
		//if (pr)
		//{
		//	printf("p1 %f %f\n", ret, Znl);
		//}
	}

	{
		float3 X, Y, Ss, Sv, Ts, Tv;

		float3 P[3];
		float t = dot(Zn, S[0]) / Znl;
		P[0] = Zn * t + S[0];
		t = dot(Zn, S[1]) / Znl;
		P[1] = Zn * t + S[1];
		t = dot(Zn, S[2]) / Znl;
		P[2] = Zn * t + S[2];

		Ss = P[0]; Sv = P[1];
		float nn = norm2(Sv - Ss);
		if (norm2(P[2] - P[0]) > nn)
		{
			Sv = P[2];
			nn = norm2(Sv - Ss);
		}
		if (norm2(P[2] - P[1]) > nn)
		{
			Ss = P[1]; Sv = P[2];
		}

		t = dot(Zn, T[0]) / Znl;
		P[0] = Zn * t + T[0];
		t = dot(Zn, T[1]) / Znl;
		P[1] = Zn * t + T[1];
		t = dot(Zn, T[2]) / Znl;
		P[2] = Zn * t + T[2];

		Ts = P[0]; Tv = P[1];
		nn = norm2(Tv - Ts);
		if (norm2(P[2] - P[0]) > nn)
		{
			Tv = P[2];
			nn = norm2(Tv - Ts);
		}
		if (norm2(P[2] - P[1]) > nn)
		{
			Ts = P[1]; Tv = P[2];
		}

		Sv -= Ss;
		Tv -= Ts;
		SegPointsGPU(X, Y, Ss, Sv, Ts, Tv);

		ret += norm2(X - Y);
		//if (pr)
		//{
		//	printf("p2 %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n", norm2(X - Y), X.x, X.y, X.z, Y.x, Y.y, Y.z, Ss.x, Ss.y, Ss.z, Sv.x, Sv.y, Sv.z, Ts.x, Ts.y, Ts.z, Tv.x, Tv.y, Tv.z);
		//}
	}
	return ret;
}


__global__ void kernel_slow_narrow_test(g_box* bvhA, g_box* bvhB, float3* vtxA, float3* vtxB, g_bvtt_SoA bvttNode, float* out_dis, float* g_minDist, int deepA, int deepB, int numTriA, int numTriB, int num)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num) return;
	int2 a_leaf;
	int2 b_leaf;
	int a, b;
	float mmin = g_minDist[0];
	{
		
		a = bvttNode.a[tid];
		b = bvttNode.b[tid];

		float newMin;
		getMinMax(bvhA[a], bvhB[b], newMin);

		if (newMin > mmin) {
			out_dis[tid] = mmin; return;
		}

		getStartAndEnd(a - (1 << (deepA - 1)), deepA, numTriA, a_leaf.x, a_leaf.y);

		getStartAndEnd(b - (1 << (deepB - 1)), deepB, numTriB, b_leaf.x, b_leaf.y);
	}
	float3 S[3];
	float3 T[3];
	//a_leaf.y = a_leaf.x + 1;
	//b_leaf.y = b_leaf.x + 1;

	for (int i = a_leaf.x; i < a_leaf.y; i++)
	{
		//g_box boxA;
		S[0] = vtxA[i * 3];
		S[1] = vtxA[i * 3 + 1];
		S[2] = vtxA[i * 3 + 2];
		//boxA.set(S[0]);
		//boxA.add(S[1]);
		//boxA.add(S[2]);
		for (int j = b_leaf.x; j < b_leaf.y; j++)
		{
			//g_box boxB;
			T[0] = vtxB[j * 3];
			T[1] = vtxB[j * 3 + 1];
			T[2] = vtxB[j * 3 + 2];
			//boxB.set(T[0]);
			//boxB.add(T[1]);
			//boxB.add(T[2]);
			float newMin;
			//getMinMax(boxA, boxB, newMin);
			
			//if (newMin >= mmin) {
			//	continue;
			//}

			float temp;
			//if (i == 1029479 && j == 1030789) {
			//	temp = raw_distance(S, T, true);
			//	//printf("raw distance %f\n", temp);
			//}
			//else {
			temp = raw_distance_simple(S, T);
			////}
			//////if (temp < mmin) mmin = temp;
			if (temp >= mmin)
			{
				continue;
			}
			//
			temp = TriDistPQP(S, T);
			//999934.375000
			//
			////float temp = min(norm2(S[0] - T[1]), norm2(S[1] - T[2]));
			if (temp < mmin) mmin = temp;
		}
	}

	//if (tid == 1000000) {
	//	printf("%d %d\n", cnt, (a_leaf.y - a_leaf.x) * (b_leaf.y - b_leaf.x));
	//}
	out_dis[tid] = mmin;
}

__global__ void kernel_findMinId(float* g_outMin, int* g_outId1, int* g_outId2, float* g_minDist, int* g_minId1, int* g_minId2, int num)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num) return;

	if (g_outMin[tid] == g_minDist[0])
	{
		g_minId1[0] = g_outId1[tid];
		g_minId2[0] = g_outId2[tid];
	}
}

__global__ void kernel_see(int* g_length, float* g_minDist)
{
	printf("%d %f\n", g_length[0], g_minDist[0]);
}

void DistanceQuery::MainProcess(int numTriA, int numTriB, int deepA, int deepB, const g_transf rigids, const float initDis, TempData* data, bool isMin, DistanceResult& result)
{
	static bool once = true;
	g_box* g_bvhA_nodes = data->get<g_box>("bvhA_nodes");
	g_box* g_bvhB_nodes = data->get<g_box>("bvhB_nodes");
	float3* g_bvhA_vtxs = data->get<float3>("bvhA_vtxs");
	float3* g_bvhB_vtxs = data->get<float3>("bvhB_vtxs");
	float3* g_bvhA_vtxsUpdate = data->get<float3>("bvhA_vtxsUpdate");
	float* g_outDis = data->getLarge<float>("outDis");
	float3* g_outP1 = data->getLarge<float3>("outP1");
	float3* g_outP2 = data->getLarge<float3>("outP2");
	int* g_mutex = data->get<int>("mutex");
	g_bvtt_SoA g_bvtt_buffer1, g_bvtt_buffer2;
	
	g_bvtt_buffer1.a = data->getLarge<int>("bvtt1_a");
	g_bvtt_buffer1.b = data->getLarge<int>("bvtt1_b");
	g_bvtt_buffer1.min = data->getLarge<float>("bvtt1_min");

	g_bvtt_buffer2.a = data->getLarge<int>("bvtt2_a");
	g_bvtt_buffer2.b = data->getLarge<int>("bvtt2_b");
	g_bvtt_buffer2.min = data->getLarge<float>("bvtt2_min");

	float* g_minDist;
	int* g_minId1;
	int* g_minId2;
	int* g_length;
	utils::malloc(g_minDist, 1);
	utils::malloc(g_minId1, 1);
	utils::malloc(g_minId2, 1);
	utils::malloc(g_length, 1);
	
	int deepNow = 1;
	
	if (isMin) {
		utils::kernel(kernel_updateVtx, numTriA * 3, 1024, g_bvhA_vtxs, g_bvhA_vtxsUpdate, rigids, numTriA * 3);
		result.times.push_back(utils::kernel("kernel_updataBV", kernel_updateBV, 1 << (deepA - 1), 128, g_bvhA_nodes, g_bvhA_vtxsUpdate, g_mutex, deepA, numTriA));
		if (once) {
			result.times.push_back(utils::kernel("kernel_updataBV_B", kernel_updateBV, 1 << (deepB - 1), 128, g_bvhB_nodes, g_bvhB_vtxs, g_mutex, deepB, numTriB));
			once = false;
		}
	}
	else
	{
		utils::kernel(kernel_updateVtx, numTriA, 1024, g_bvhA_vtxs, g_bvhA_vtxsUpdate, rigids, numTriA);
		result.times.push_back(utils::kernel("kernel_updataBV", kernel_updateBV_ForMax, 1 << (deepA - 1), 128, g_bvhA_nodes, g_bvhA_vtxsUpdate, g_mutex, deepA, numTriA));
		if (once) {
			result.times.push_back(utils::kernel("kernel_updataBV_B", kernel_updateBV_ForMax, 1 << (deepB - 1), 128, g_bvhB_nodes, g_bvhB_vtxs, g_mutex, deepB, numTriB));
			once = false;
		}
	}


	
	//utils::kernel(kernel_updateBV, 1 << (deepA - 1), 128, g_bvhA_nodes, g_bvhA_vtxsUpdate, g_mutex, deepA, numTriA);
	
	if (isMin) {
		result.times.push_back(utils::kernel("kernel_gpu_init", kernel_gpu_init, 1, 1, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_length, g_length, result.id1, result.id2, g_minDist, initDis));
	}
	else {
		result.times.push_back(utils::kernel("kernel_gpu_init", kernel_gpu_init_formax, 1, 1, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_length, g_length, g_minDist, initDis));
	}
	utils::memset(g_length, 1);
	int length_cpu = 1;
	
	int proDeep = calProDeep(deepA, deepB, 1, deepNow);
	result.proDeeps.push_back(proDeep);
	result.bvtts.push_back(1);
	printf("%d\n", proDeep);
	if (isMin)
	{
		result.times.push_back(utils::kernel("kernel_fast_traversal", kernel_fast_traversal, (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, 1, proDeep));
	}
	else {
		result.times.push_back(utils::kernel("kernel_fast_traversal", kernel_fast_traversal_formax, (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, 1, proDeep));
	}

	cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", length_cpu);
	//result.times.push_back(utils::minReduce("cub reduce bvtt\n", g_bvtt_buffer2.max, g_minDist, length_cpu));
	bool flag = false;
	bool earlyExit = false;
	deepNow += proDeep;
	int deepMax = max(deepA, deepB);
	while (deepNow < deepMax)
	{
		utils::memset(g_length, 1);
		proDeep = calProDeep(deepA, deepB, length_cpu, deepNow);
		result.proDeeps.push_back(proDeep);
		result.bvtts.push_back(length_cpu);
		printf("%d %d %d\n", length_cpu, deepNow, proDeep);
		utils::memcpy(&result.min, g_minDist, 1, cudaMemcpyDeviceToHost);
		printf("now min %f\n", result.min);
		if (deepNow < deepA && deepNow < deepB) {
			if (flag)
			{
				if (isMin) {
					result.times.push_back(utils::kernel("kernel_fast_traversal", kernel_fast_traversal,
						length_cpu * (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, length_cpu, proDeep));
				}
				else {
					result.times.push_back(utils::kernel("kernel_fast_traversal", kernel_fast_traversal_formax,
						length_cpu * (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, length_cpu, proDeep));
				}
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				//result.times.push_back(utils::minReduce("cub reduce bvtt", g_bvtt_buffer2.max, g_minDist, length_cpu));
				utils::memset(g_length, 1);
				flag = false;
			}
			else {
				if (isMin) {
					result.times.push_back(utils::kernel("kernel_fast_traversal", kernel_fast_traversal,
						length_cpu * (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer2, g_bvtt_buffer1, g_length, g_minDist, length_cpu, proDeep));
				}
				else {
					result.times.push_back(utils::kernel("kernel_fast_traversal", kernel_fast_traversal_formax,
						length_cpu * (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer2, g_bvtt_buffer1, g_length, g_minDist, length_cpu, proDeep));
				}
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				//result.times.push_back(utils::minReduce("cub reduce bvtt\n", g_bvtt_buffer1.max, g_minDist, length_cpu));
				utils::memset(g_length, 1);
				flag = true;
			}
		}
		else {
			if (flag)
			{
				result.times.push_back(utils::kernel("kernel_fast_traversal_onlyB", kernel_fast_traversal_onlyB, 
					length_cpu * (1 << proDeep), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepA, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, length_cpu, proDeep));
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				utils::memset(g_length, 1);
				flag = false;
			}
			else {
				result.times.push_back(utils::kernel("kernel_fast_traversal_onlyB", kernel_fast_traversal_onlyB, 
					length_cpu * (1 << proDeep), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepA, deepNow, g_bvtt_buffer2, g_bvtt_buffer1, g_length, g_minDist, length_cpu, proDeep));
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				utils::memset(g_length, 1);
				flag = true;
			}
		}
		deepNow += proDeep;

		if (length_cpu > (1 << 23))
		{
			earlyExit = true;
			break;
		}
	}

	printf("%d\n", length_cpu);
	if (earlyExit)
	{
		if (flag) {
			result.times.push_back(utils::kernel("kernel_slow_narrow_test", kernel_slow_narrow_test, length_cpu, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_outDis, g_minDist, deepNow, deepNow, numTriA, numTriB, length_cpu));
		}
		else {
			result.times.push_back(utils::kernel("kernel_slow_narrow_test", kernel_slow_narrow_test, length_cpu, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer2, g_outDis, g_minDist, deepNow, deepNow, numTriA, numTriB, length_cpu));
		}
		result.times.push_back(utils::minReduce("name", g_outDis, g_minDist, length_cpu));
	}
	else {
		if (flag)
		{
			if (isMin) {
				result.times.push_back(utils::kernel("kernel_narrow_test", kernel_narrow_test, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_outDis, g_minId1, g_minId2, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4));
			}
			else {
				result.times.push_back(utils::kernel("kernel_narrow_test_formax", kernel_narrow_test_formax, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_outDis, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4));
			}
		}
		else {
			if (isMin) {
				result.times.push_back(utils::kernel("kernel_narrow_test", kernel_narrow_test, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer2, g_outDis, g_minId1, g_minId2, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4));
			}
			else {
				result.times.push_back(utils::kernel("kernel_narrow_test_formax", kernel_narrow_test_formax, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer2, g_outDis, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4));
			}
		}

		if (isMin) {
			result.times.push_back(utils::minReduce("name", g_outDis, g_minDist, length_cpu * 4));
		}
		else {
			result.times.push_back(utils::maxReduce("name", g_outDis, g_minDist, length_cpu * 4));
		}
	}
	
	

	utils::memcpy(&result.min, g_minDist, 1, cudaMemcpyDeviceToHost);
	utils::memcpy(&result.id1, g_minId1, 1, cudaMemcpyDeviceToHost);
	utils::memcpy(&result.id2, g_minId2, 1, cudaMemcpyDeviceToHost);
	cudaFree(g_minDist);
	cudaFree(g_length);
	cudaFree(g_minId1);
	cudaFree(g_minId2);
	getLastCudaError("kernel_RigidRigidDistance");

	result.sum_time = 0;
	for (auto t : result.times)
	{
		result.sum_time += t;
	}
}

#include <chrono>

void DistanceQuery::MainProcessAllTime(int numTriA, int numTriB, int deepA, int deepB, const g_transf rigids, const float initDis, TempData* data, bool isMin, DistanceResult& result)
{
	static bool once = true;
	g_box* g_bvhA_nodes = data->get<g_box>("bvhA_nodes");
	g_box* g_bvhB_nodes = data->get<g_box>("bvhB_nodes");
	float3* g_bvhA_vtxs = data->get<float3>("bvhA_vtxs");
	float3* g_bvhB_vtxs = data->get<float3>("bvhB_vtxs");
	float3* g_bvhA_vtxsUpdate = data->get<float3>("bvhA_vtxsUpdate");
	float* g_outDis = data->getLarge<float>("outDis");
	int* g_mutex = data->get<int>("mutex");
	g_bvtt_SoA g_bvtt_buffer1, g_bvtt_buffer2;

	g_bvtt_buffer1.a = data->getLarge<int>("bvtt1_a");
	g_bvtt_buffer1.b = data->getLarge<int>("bvtt1_b");
	g_bvtt_buffer1.min = data->getLarge<float>("bvtt1_min");

	g_bvtt_buffer2.a = data->getLarge<int>("bvtt2_a");
	g_bvtt_buffer2.b = data->getLarge<int>("bvtt2_b");
	g_bvtt_buffer2.min = data->getLarge<float>("bvtt2_min");

	float* g_minDist;
	int* g_minId1;
	int* g_minId2;
	int* g_length;
	utils::malloc(g_minDist, 1);
	utils::malloc(g_minId1, 1);
	utils::malloc(g_minId2, 1);
	utils::malloc(g_length, 1);

	int deepNow = 1;

	utils::kernel(kernel_updateVtx, numTriA * 3, 1024, g_bvhA_vtxs, g_bvhA_vtxsUpdate, rigids, numTriA * 3);

	
	utils::kernel(kernel_updateBV, 1 << (deepA - 1), 128, g_bvhA_nodes, g_bvhA_vtxsUpdate, g_mutex, deepA, numTriA);
	//utils::kernel(kernel_updateBV, 1 << (deepA - 1), 128, g_bvhA_nodes, g_bvhA_vtxsUpdate, g_mutex, deepA, numTriA);
	if (once) {
		utils::kernel(kernel_updateBV, 1 << (deepB - 1), 128, g_bvhB_nodes, g_bvhB_vtxs, g_mutex, deepB, numTriB);
		once = false;
	}

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	utils::kernel(kernel_gpu_init, 1, 1, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_length, g_length, result.id1, result.id2, g_minDist, initDis);
	
	

	utils::memset(g_length, 1);
	int length_cpu = 1;

	int proDeep = calProDeep(deepA, deepB, 1, deepNow);
	utils::kernel(kernel_fast_traversal, (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, 1, proDeep);
	cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
	//result.times.push_back(utils::minReduce("cub reduce bvtt\n", g_bvtt_buffer2.max, g_minDist, length_cpu));
	bool flag = false;
	bool earlyExit = false;
	deepNow += proDeep;
	int deepMax = max(deepA, deepB);
	while (deepNow < deepMax)
	{
		utils::memset(g_length, 1);
		proDeep = calProDeep(deepA, deepB, length_cpu, deepNow);
		if (deepNow < deepA && deepNow < deepB) {
			if (flag)
			{
				utils::kernel(kernel_fast_traversal,
					length_cpu * (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, length_cpu, proDeep);
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				//result.times.push_back(utils::minReduce("cub reduce bvtt", g_bvtt_buffer2.max, g_minDist, length_cpu));
				utils::memset(g_length, 1);
				flag = false;
			}
			else {
				utils::kernel(kernel_fast_traversal,
					length_cpu * (1 << (proDeep << 1)), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepNow, g_bvtt_buffer2, g_bvtt_buffer1, g_length, g_minDist, length_cpu, proDeep);
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				//result.times.push_back(utils::minReduce("cub reduce bvtt\n", g_bvtt_buffer1.max, g_minDist, length_cpu));
				utils::memset(g_length, 1);
				flag = true;
			}
		}
		else {
			if (flag)
			{
				utils::kernel(kernel_fast_traversal_onlyB,
					length_cpu * (1 << proDeep), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepA, deepNow, g_bvtt_buffer1, g_bvtt_buffer2, g_length, g_minDist, length_cpu, proDeep);
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				utils::memset(g_length, 1);
				flag = false;
			}
			else {
				utils::kernel(kernel_fast_traversal_onlyB,
					length_cpu * (1 << proDeep), 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, numTriA, numTriB, deepA, deepNow, g_bvtt_buffer2, g_bvtt_buffer1, g_length, g_minDist, length_cpu, proDeep);
				cudaMemcpy(&length_cpu, g_length, sizeof(int), cudaMemcpyDeviceToHost);
				utils::memset(g_length, 1);
				flag = true;
			}
		}
		deepNow += proDeep;

		if (length_cpu > (1 << 23))
		{
			earlyExit = true;
			break;
		}
	}
	printf("cpu length %d\n", length_cpu);
	if (earlyExit)
	{
		if (flag) {
			utils::kernel(kernel_slow_narrow_test, length_cpu, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_outDis, g_minDist, deepNow, deepNow, numTriA, numTriB, length_cpu);
		}
		else {
			utils::kernel(kernel_slow_narrow_test, length_cpu, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer2, g_outDis, g_minDist, deepNow, deepNow, numTriA, numTriB, length_cpu);
		}
		utils::minReduce(g_outDis, g_minDist, length_cpu);
	}
	else {
		if (flag)
		{
			utils::kernel(kernel_narrow_test, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_outDis, g_minId1, g_minId2, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4);
		}
		else {
			utils::kernel(kernel_narrow_test, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer2, g_outDis, g_minId1, g_minId2, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4);
		}
		utils::minReduce(g_outDis, g_minDist, length_cpu * 4);
	}
	utils::memcpy(&result.min, g_minDist, 1, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	//printf("%s time: %f ms\n", name.c_str(), elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (flag)
	{
		utils::kernel(kernel_findId, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer1, g_outDis, g_minId1, g_minId2, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4);
	}
	else {
		utils::kernel(kernel_findId, length_cpu * 4, 128, g_bvhA_nodes, g_bvhB_nodes, g_bvhA_vtxsUpdate, g_bvhB_vtxs, g_bvtt_buffer2, g_outDis, g_minId1, g_minId2, g_minDist, deepA, deepB, numTriA, numTriB, length_cpu * 4);
	}

	utils::memcpy(&result.id1, g_minId1, 1, cudaMemcpyDeviceToHost);
	utils::memcpy(&result.id2, g_minId2, 1, cudaMemcpyDeviceToHost);

	
	cudaFree(g_minDist);
	cudaFree(g_length);
	cudaFree(g_minId1);
	cudaFree(g_minId2);
	getLastCudaError("kernel_RigidRigidDistance");

	result.sum_time = elapsedTime;
}

//--------------------------------------------------------------------------
// SegPoints() 
//
// Returns closest points between an segment pair.
// Implemented from an algorithm described in
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.   
//--------------------------------------------------------------------------
__host__ __device__ void
SegPoints(float3& VEC,
	float3& X, float3& Y, // closest points
	const float3& P, const float3& A, // seg 1 origin, vector
	const float3& Q, const float3& B); // seg 2 origin, vector

//--------------------------------------------------------------------------
// TriDist() 
//
// Computes the closest points on two triangles, and returns the 
// distance between them.
// 
// S and T are the triangles, stored tri[point][dimension].
//
// If the triangles are disjoint, P and Q give the closest points of 
// S and T respectively. However, if the triangles overlap, P and Q 
// are basically a random pair of points from the triangles, not 
// coincident points on the intersection of the triangles, as might 
// be expected.
//--------------------------------------------------------------------------



__host__ __device__ float
triDist(
	const float3& P1, const float3& P2, const float3& P3,
	const float3& Q1, const float3& Q2, const float3& Q3,
	float3& rP, float3& rQ);

#define VmV(T, Q, P)  {T=Q-P;}
#define VpVxS(Vr, V1, V2, s) { Vr = V1+V2*s; }
#define VdotV(A, B) dot(A, B)
#define VcV(A, B) { A=B; }
#define VpV(T, Q, P)  {T=Q+P;}
#define VcrossV(A, B, C) { A = cross(B, C);}
#define VxS(A, B, s) { A = B*s; }
#define VdistV2(A, B) norm2(A-B)

__inline__ __host__ __device__ void
SegPointsGPU(float3& X, float3& Y, // closest points
	const float3& P, const float3& A, // seg 1 origin, vector
	const float3& Q, const float3& B) // seg 2 origin, vector
{
	float3 T, TMP;
	float A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;

	VmV(T, Q, P);
	A_dot_A = VdotV(A, A);
	B_dot_B = VdotV(B, B);
	A_dot_B = VdotV(A, B);
	A_dot_T = VdotV(A, T);
	B_dot_T = VdotV(B, T);

	// t parameterizes ray P,A 
	// u parameterizes ray Q,B 

	float t, u;

	// compute t for the closest point on ray P,A to
	// ray Q,B

	float denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

	t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

	// clamp result so t is on the segment P,A

	if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;

	// find u for point on ray Q,B closest to point at t

	u = (t * A_dot_B - B_dot_T) / B_dot_B;

	// if u is on segment Q,B, t and u correspond to 
	// closest points, otherwise, clamp u, recompute and
	// clamp t 

	if ((u <= 0) || isnan(u)) {
		VcV(Y, Q);
		t = A_dot_T / A_dot_A;
	}
	else if (u >= 1) {
		VpV(Y, Q, B);
		t = (A_dot_B + A_dot_T) / A_dot_A;
	}
	else {
		VpVxS(Y, Q, B, u);
	}

	if ((t <= 0) || isnan(t)) {
		VcV(X, P);
	}
	else if (t >= 1) {
		VpV(X, P, A);
	}
	else {
		VpVxS(X, P, A, t);
	}
}

__inline__ __host__ __device__ void
SegPoints(float3& VEC,
	float3& X, float3& Y, // closest points
	const float3& P, const float3& A, // seg 1 origin, vector
	const float3& Q, const float3& B) // seg 2 origin, vector
{
	float3 T, TMP;
	float A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;

	VmV(T, Q, P);
	A_dot_A = VdotV(A, A);
	B_dot_B = VdotV(B, B);
	A_dot_B = VdotV(A, B);
	A_dot_T = VdotV(A, T);
	B_dot_T = VdotV(B, T);

	// t parameterizes ray P,A 
	// u parameterizes ray Q,B 

	float t, u;

	// compute t for the closest point on ray P,A to
	// ray Q,B

	float denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

	t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

	// clamp result so t is on the segment P,A

	if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;

	// find u for point on ray Q,B closest to point at t

	u = (t * A_dot_B - B_dot_T) / B_dot_B;

	// if u is on segment Q,B, t and u correspond to 
	// closest points, otherwise, clamp u, recompute and
	// clamp t 

	if ((u <= 0) || isnan(u)) {

		VcV(Y, Q);

		t = A_dot_T / A_dot_A;

		if ((t <= 0) || isnan(t)) {
			VcV(X, P);
			VmV(VEC, Q, P);
		}
		else if (t >= 1) {
			VpV(X, P, A);
			VmV(VEC, Q, X);
		}
		else {
			VpVxS(X, P, A, t);
			VcrossV(TMP, T, A);
			VcrossV(VEC, A, TMP);
		}
	}
	else if (u >= 1) {

		VpV(Y, Q, B);

		t = (A_dot_B + A_dot_T) / A_dot_A;

		if ((t <= 0) || isnan(t)) {
			VcV(X, P);
			VmV(VEC, Y, P);
		}
		else if (t >= 1) {
			VpV(X, P, A);
			VmV(VEC, Y, X);
		}
		else {
			VpVxS(X, P, A, t);
			VmV(T, Y, P);
			VcrossV(TMP, T, A);
			VcrossV(VEC, A, TMP);
		}
	}
	else {

		VpVxS(Y, Q, B, u);

		if ((t <= 0) || isnan(t)) {
			VcV(X, P);
			VcrossV(TMP, T, B);
			VcrossV(VEC, B, TMP);
		}
		else if (t >= 1) {
			VpV(X, P, A);
			VmV(T, Q, X);
			VcrossV(TMP, T, B);
			VcrossV(VEC, B, TMP);
		}
		else {
			VpVxS(X, P, A, t);
			VcrossV(VEC, A, B);
			if (VdotV(VEC, T) < 0) {
				VxS(VEC, VEC, -1);
			}
		}
	}
}

__host__ __device__ float
TriDistPQP(const float3* S, const float3* T)
{
	// Compute vectors along the 6 sides

	//float3 Sv[3], Tv[3], VEC;
	float3 P, Q;


	// For each edge pair, the vector connecting the closest points 
	// of the edges defines a slab (parallel planes at head and tail
	// enclose the slab). If we can show that the off-edge vertex of 
	// each triangle is outside of the slab, then the closest points
	// of the edges are the closest points for the triangles.
	// Even if these tests fail, it may be helpful to know the closest
	// points found, and whether the triangles were shown disjoint


	float mindd;
	int shown_disjoint = 0;

	mindd = VdistV2(S[0], T[0]) + 1;  // Set first minimum safely high

	for (int i = 0; i < 3; i++)
	{
		float3 tSv = S[(i + 1) % 3] - S[i];
		for (int j = 0; j < 3; j++)
		{
			// Find closest points on edges i & j, plus the 
			// vector (and distance squared) between these points
			float3 tTv = T[(j + 1) % 3] - T[j];
			float3 V, Z;
			SegPointsGPU(P, Q, S[i], tSv, T[j], tTv);

			VmV(V, Q, P);
			float dd = VdotV(V, V);

			// Verify this closest point pair only if the distance 
			// squared is less than the minimum found thus far.

			if (dd <= mindd)
			{
				mindd = dd;

				VmV(Z, S[(i + 2) % 3], P);
				float a = VdotV(Z, V);
				VmV(Z, T[(j + 2) % 3], Q);
				float b = VdotV(Z, V);

				if ((a <= 0) && (b >= 0))
					return dd;

				if (a < 0) a = 0;
				if (b > 0) b = 0;
				if ((dd - a + b) > 0) shown_disjoint = 1;
			}
		}
	}

	// No edge pairs contained the closest points.  
	// either:
	// 1. one of the closest points is a vertex, and the
	//    other point is interior to a face.
	// 2. the triangles are overlapping.
	// 3. an edge of one triangle is parallel to the other's face. If
	//    cases 1 and 2 are not true, then the closest points from the 9
	//    edge pairs checks above can be taken as closest points for the
	//    triangles.
	// 4. possibly, the triangles were degenerate.  When the 
	//    triangle points are nearly colinear or coincident, one 
	//    of above tests might fail even though the edges tested
	//    contain the closest points.

	// First check for case 1

	{
		float3 Sn;
		float Snl;
		VcrossV(Sn, S[1] - S[0], S[2] - S[1]); // Compute normal to S triangle
		Snl = VdotV(Sn, Sn);      // Compute square of length of normal

		// If cross product is long enough,

		if (Snl > 1e-15)
		{
			// Get projection lengths of T points

			float Tp[3];

			Tp[0] = VdotV(S[0] - T[0], Sn);
			Tp[1] = VdotV(S[0] - T[1], Sn);
			Tp[2] = VdotV(S[0] - T[2], Sn);

			// If Sn is a separating direction,
			// find point with smallest projection

			int point = -1;
			if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
			{
				if (Tp[0] < Tp[1]) point = 0; else point = 1;
				if (Tp[2] < Tp[point]) point = 2;
			}
			else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
			{
				if (Tp[0] > Tp[1]) point = 0; else point = 1;
				if (Tp[2] > Tp[point]) point = 2;
			}

			// If Sn is a separating direction, 

			if (point >= 0)
			{
				shown_disjoint = 1;

				// Test whether the point found, when projected onto the 
				// other triangle, lies within the face.

				if (VdotV(T[point] - S[0], cross(Sn, S[1] - S[0])) > 0 && VdotV(T[point] - S[1], cross(Sn, S[2] - S[1])) > 0 && VdotV(T[point] - S[2], cross(Sn, S[0] - S[2])) > 0)
				{
					// T[point] passed the test - it's a closest point for 
					// the T triangle; the other point is on the face of S

					//return sqrt(VdistV2(P, Q));
					return norm2(Sn * (Tp[point] / Snl));
					//return VdistV2(P, Q);
				}
			}
		}
	}

	{
		float3 Tn;
		float Tnl;
		VcrossV(Tn, T[1] - T[0], T[2] - T[1]);
		Tnl = VdotV(Tn, Tn);

		if (Tnl > 1e-15)
		{
			float Sp[3];

			Sp[0] = VdotV(T[0] - S[0], Tn);

			Sp[1] = VdotV(T[0] - S[1], Tn);

			Sp[2] = VdotV(T[0] - S[2], Tn);

			int point = -1;
			if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
			{
				if (Sp[0] < Sp[1]) point = 0; else point = 1;
				if (Sp[2] < Sp[point]) point = 2;
			}
			else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
			{
				if (Sp[0] > Sp[1]) point = 0; else point = 1;
				if (Sp[2] > Sp[point]) point = 2;
			}

			if (point >= 0)
			{
				shown_disjoint = 1;
				if (VdotV(S[point] - T[0], cross(Tn, T[1] - T[0])) > 0 && VdotV(S[point] - T[1], cross(Tn, T[2] - T[1])) > 0 && VdotV(S[point] - T[2], cross(Tn, T[0] - T[2])) > 0)
				{
					//VcV(P, S[point]);
					//VpVxS(Q, S[point], Tn, Sp[point] / Tnl);
					//return sqrt(VdistV2(P, Q));
					return norm2(Tn * (Sp[point] / Tnl));
				}
			}
		}
	}
	// Case 1 can't be shown.
	// If one of these tests showed the triangles disjoint,
	// we assume case 3 or 4, otherwise we conclude case 2, 
	// that the triangles overlap.

	if (shown_disjoint)
	{
		//return sqrt(mindd);
		return mindd;
	}
	else return 0;
}
