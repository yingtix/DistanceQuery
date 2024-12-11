#include "TempData.cuh"
#include "utils.cuh"

TempData::TempData()
{
	used = new bool[size];
	mem = nullptr;
	utils::malloc(mem, MAX_THING * size);
	for (int i = 0; i < size; i++) {
		used[i] = false;
	}

	utils::malloc(count, 1);
}

int* TempData::useCount()
{
	utils::memset(count, 1);
	return count;
}

int TempData::getCount()
{
	return utils::getValue(count, 0);
}
TempData::~TempData()
{
	cudaFree(mem);

	for (auto it : Mlarge)
	{
		cudaFree(it.second);
	}
}


