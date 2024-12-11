#pragma once
#include "CudaBase/helper_math.h"
#include <iostream>

bool readobjfile(const std::string& path, int& numTri, float3*& vtxs, bool isMin, float scale = 1)
{
	vector<int3> triset;
	vector<float3> vtxset;

	FILE* fp = fopen(path.c_str(), "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
			double x, y, z;
			sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);
			vtxset.push_back(make_float3(x, y, z) * scale);
		}
		else if (buf[0] == 'v' && buf[1] == 't') {
			double x, y;
			sscanf(buf + 3, "%lf%lf", &x, &y);
		} else if (buf[0] == 'f' && buf[1] == ' ') {
			int id0, id1, id2, id3 = 0;
			int tid0, tid1, tid2, tid3 = 0;
			bool quad = false;

			int count = sscanf(buf + 2, "%d/%d", &id0, &tid0);
			char* nxt = strchr(buf + 2, ' ');
			sscanf(nxt + 1, "%d/%d", &id1, &tid1);
			nxt = strchr(nxt + 1, ' ');
			sscanf(nxt + 1, "%d/%d", &id2, &tid2);

			nxt = strchr(nxt + 1, ' ');
			if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
				if (sscanf(nxt + 1, "%d/%d", &id3, &tid3))
					quad = true;
			}

			id0--, id1--, id2--, id3--;
			tid0--, tid1--, tid2--, tid3--;

			triset.push_back(make_int3(id0, id1, id2));

			if (quad) {
				triset.push_back(make_int3(id0, id2, id3));
			}
		}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	float3 minp = vtxset[0];
	float3 maxp = vtxset[0];
	for (int i = 0; i < vtxset.size(); i++)
	{
		minp = fminf(minp, vtxset[i]);
		maxp = fmaxf(maxp, vtxset[i]);
	}

	printf("%f %f %f\n", minp.x, minp.y, minp.z);
	printf("%f %f %f\n", maxp.x, maxp.y, maxp.z);


	//nrm = new float3[vtxset.size()];
	//memset(nrm, 0, sizeof(float3) * vtxset.size());
	//for (uint32_t i = 0; i < triset.size(); i++) {
	//	auto vid1 = (triset)[i].x;
	//	auto vid2 = (triset)[i].y;
	//	auto vid3 = (triset)[i].z;
	//	auto vertex1 = vtxset[vid1];
	//	auto vertex2 = vtxset[vid2];
	//	auto vertex3 = vtxset[vid3];
	//
	//	auto face_normal = normalize(cross(vertex2 - vertex1, vertex3 - vertex1));
	//
	//	nrm[vid1] += face_normal;
	//	nrm[vid2] += face_normal;
	//	nrm[vid3] += face_normal;
	//}
	//for (uint32_t i = 0; i < vtxset.size(); i++) {
	//	nrm[i] = normalize(nrm[i]);
	//}

	if (isMin) {
		numTri = triset.size();
		vtxs = new float3[numTri * 3];
		for (int i = 0; i < numTri; i++)
		{
			const auto& tri = triset[i];
			vtxs[i * 3] = vtxset[tri.x];
			vtxs[i * 3 + 1] = vtxset[tri.y];
			vtxs[i * 3 + 2] = vtxset[tri.z];
		}
	}
	else {
		numTri = vtxset.size();
		vtxs = new float3[numTri];
		for (int i = 0; i < numTri; i++)
		{
			vtxs[i] = vtxset[i];
		}
	}
	return true;
}