#pragma once
#include "Application.h"

class TempData;

struct float3;
class DistanceQueryApp : public Application
{
public:
	TempData* gpu_data;

	int deepA, deepB;
	int numTriA, numTriB;

	float3* vtxsA;
	float3* vtxsB;
	float3* sort_vtxA;
	float3* sort_vtxB;

	float ptA[3] = { 0.0f, 0.0f, 0.0f };
	float ptB[3] = { 0.0f, 0.0f, 0.0f };

	void getResultTriID(int& id0, int& id1);

	int benchmarkId = 0;
	int calculateMinD = true;
	float minDist = 0.0f;
public:
	virtual void run() override;
	virtual void step() override;
	virtual void init(int benckmarkId, bool calculateMinD, std::string path) override;
	virtual void end() override;
};