#include "MainCodeBuildBVH.cuh"
#include "utils.cuh"
#include "Type.cuh"
#include "TempData.cuh"

#include <algorithm>

void reorder(int x, int l, int mid, int r, int* vis, int* buffer, int* id)
{
	for (int i = l; i <= r; i++)
	{
		buffer[i] = id[i];
	}
	int p1 = l;
	int p2 = mid + 1;
	for (int i = l; i <= r; i++)
	{
		if (vis[buffer[i]] == x)
		{
			id[p1] = buffer[i];
			p1++;
		}
		else {
			id[p2] = buffer[i];
			p2++;
		}
	}
}

void build(int x, int deep, const int MaxDeep, int l, int r, int* vis, int* buffer, float* xc, float* yc, float* zc, int* xid, int* yid, int* zid, int2* leafs)
{
	if (deep == MaxDeep) {
		leafs[x - (1 << (deep - 1))].x = l;
		leafs[x - (1 << (deep - 1))].y = r + 1;
		return;
	}
	float dx = xc[xid[r]] - xc[xid[l]];
	float dy = yc[yid[r]] - yc[yid[l]];
	float dz = zc[zid[r]] - zc[zid[l]];

	int* id;
	int* other1, * other2;
	if (dx > dy && dx > dz)
	{
		id = xid;
		other1 = yid;
		other2 = zid;
	}
	else if (dy > dx && dy > dz)
	{
		id = yid;
		other1 = xid;
		other2 = zid;
	}
	else {
		id = zid;
		other1 = xid;
		other2 = yid;
	}
	int mid = (l + r) / 2;
	for (int i = l; i <= mid; i++)
	{
		vis[id[i]] = x;
	}

	reorder(x, l, mid, r, vis, buffer, other1);
	reorder(x, l, mid, r, vis, buffer, other2);
	build(x * 2, deep + 1, MaxDeep, l, mid, vis, buffer, xc, yc, zc, xid, yid, zid, leafs);
	build(x * 2 + 1, deep + 1, MaxDeep, mid + 1, r, vis, buffer, xc, yc, zc, xid, yid, zid, leafs);
}


void BuildBVH::MainProcess(int numTri, float3* vtxs, int& ret_deep, TempData* gpu_data, bool isA, bool isMin, float3*&sort_vtxs)
{
	int temp = numTri;
	int deep = 0;
	while (temp)
	{
		deep++;
		temp /= 2;
	}
	printf("%d %d\n", deep, numTri);
	printf("sort\n");
	ret_deep = deep;
	int* xid = new int[numTri];
	int* yid = new int[numTri];
	int* zid = new int[numTri];
	float* xc = new float[numTri];
	float* yc = new float[numTri];
	float* zc = new float[numTri];
	printf("sort\n");
	for (int i = 0; i < numTri; i++) xid[i] = yid[i] = zid[i] = i;

	for (int i = 0; i < numTri; i++) {
		float3 center;
		if (isMin) {
			float3 p1 = vtxs[i * 3];
			float3 p2 = vtxs[i * 3 + 1];
			float3 p3 = vtxs[i * 3 + 2];

			g_box box;
			box.set(p1);
			box.add(p2);
			box.add(p3);
			center = box.center();
		}
		else {
			center = vtxs[i];
		}
		xc[i] = center.x;
		yc[i] = center.y;
		zc[i] = center.z;
	}

	std::sort(xid, xid + numTri, [xc](int a, int b) {return xc[a] < xc[b]; });
	std::sort(yid, yid + numTri, [yc](int a, int b) {return yc[a] < yc[b]; });
	std::sort(zid, zid + numTri, [zc](int a, int b) {return zc[a] < zc[b]; });
	printf("sort\n");
	int2* leafs = new int2[(1 << (deep - 1))];
	if (isMin) {
		sort_vtxs = new float3[numTri * 3];
	}
	else {
		sort_vtxs = new float3[numTri * 3];
	}

	int* vis = new int[numTri * 2];
	memset(vis, 0, sizeof(int) * numTri * 2);
	int* buffer = new int[numTri * 2];
	printf("build\n");
	build(1, 1, deep, 0, numTri - 1, vis, buffer, xc, yc, zc, xid, yid, zid, leafs);
	for (int i = 0; i < numTri; i++)
	{
		if (isMin) {
			auto tri = xid[i];
			sort_vtxs[i * 3] = vtxs[tri * 3];
			sort_vtxs[i * 3 + 1] = vtxs[tri * 3 + 1];
			sort_vtxs[i * 3 + 2] = vtxs[tri * 3 + 2];
		}
		else {
			auto tri = xid[i];
			sort_vtxs[i] = vtxs[tri];
		}
	}
	delete[] vis;
	delete[] buffer;
	delete[] xid;
	delete[] yid;
	delete[] zid;
	delete[] xc;
	delete[] yc;
	delete[] zc;
	printf("sort\n");
	g_box* g_bvh_nodes;
	float3* g_bvh_vtxs;
	if (isA)
	{
		g_bvh_vtxs = gpu_data->get<float3>("bvhA_vtxs");
	}
	else {
		g_bvh_vtxs = gpu_data->get<float3>("bvhB_vtxs");
	}
	printf("sort\n");

	if (isMin) {
		utils::memcpy(g_bvh_vtxs, sort_vtxs, numTri * 3);
	}
	else {
		utils::memcpy(g_bvh_vtxs, sort_vtxs, numTri);
	}
	delete[] leafs;
}
