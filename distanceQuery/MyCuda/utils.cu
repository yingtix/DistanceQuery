#include "utils.cuh"
#include <cub/cub.cuh>

size_t utils::cubTempStorageBytes = 0;
void* utils::cubTempStorage = nullptr;

static void* cubTempStorage;
static size_t cubTempStorageBytes;


void utils::allocCubTemp()
{
	printf("CUB use mem: %d bytes\n", cubTempStorageBytes);
	checkCudaErrors(cudaMalloc((void**)&cubTempStorage, cubTempStorageBytes * sizeof(int)));
}

template<typename T>
void utils::sortRegist(int* label, int* label_out, int num_items)
{
	T* d_in;
	T* d_out;
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(cubTempStorage, cubTempStorageBytes, d_in, d_out, label, label_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::sort(T* d_in, T* d_out, int* label, int* label_out, int num_items)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cub::DeviceRadixSort::SortPairs(cubTempStorage, cubTempStorageBytes, d_in, d_out, label, label_out, num_items);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%s time: %f ms\n", "sort", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

template<typename T>
void utils::minReduceRegist(int num_items) {
	T* d_in = nullptr;
	T* d_out = nullptr;
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceReduce::Min(storage, storage_bytes, d_in, d_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::maxReduceRegist(int num_items) {
	T* d_in = nullptr;
	T* d_out = nullptr;
	void* storage = nullptr;
	size_t storage_bytes = 0;
	cub::DeviceReduce::Max(storage, storage_bytes, d_in, d_out, num_items);
	if (storage_bytes > cubTempStorageBytes) cubTempStorageBytes = storage_bytes;
}

template<typename T>
void utils::minReduce(T* d_in, T* d_out, int num_items)
{
	cub::DeviceReduce::Min(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);
}

template<typename T>
void utils::maxReduce(T* d_in, T* d_out, int num_items)
{
	cub::DeviceReduce::Max(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);
}

template<typename T>
float utils::minReduce(const std::string& name, T* d_in, T* d_out, int num_items)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	printf("reduce %d\n", num_items);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cub::DeviceReduce::Min(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%s time: %f ms\n", "min reduce", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime;
}

template<typename T>
float utils::maxReduce(const std::string& name, T* d_in, T* d_out, int num_items)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	printf("reduce %d\n", num_items);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cub::DeviceReduce::Max(cubTempStorage, cubTempStorageBytes, d_in, d_out, num_items);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%s time: %f ms\n", "min reduce", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime;
}

template void utils::minReduce<float>(float*, float*, int);
template void utils::maxReduce<float>(float*, float*, int);
template float utils::minReduce<float>(const std::string&, float*, float*, int);
template float utils::maxReduce<float>(const std::string&, float*, float*, int);
template void utils::minReduceRegist<float>(int);
template void utils::maxReduceRegist<float>(int);