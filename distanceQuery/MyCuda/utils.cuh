#pragma once
#include <cuda.h>

#include "cuda_runtime.h"
#include "cudart_platform.h"
#include "device_launch_parameters.h"
#include "CudaBase/helper_cuda.h"

//#include "device/device_scan.cuh"

struct CustomLess
{
	__device__ bool operator()(const int2& lhs, const int2& rhs)
	{
		return (lhs.x < rhs.x) || (lhs.x == rhs.x && lhs.y < rhs.y);
	}
};

#include <iostream>
class utils {
public:
	template<typename FUNC, typename... ARGS>
	static inline void kernel(FUNC func, unsigned int a, unsigned int b, ARGS... args) {
		func << <(a + b - 1) / b, b >> > (args...);
		//getLastCudaError("name");
	}

	template<typename FUNC, typename... ARGS>
	static inline float kernel(const std::string& name, FUNC func, unsigned int a, unsigned int b, ARGS... args) {
		cudaEvent_t start, stop;
		float elapsedTime = 0.0;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		func << <(a + b - 1) / b, b >> > (args...);
		getLastCudaError(name.c_str());

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		std::cout << name << " time: " << elapsedTime << " ms\n";
		//printf("%s time: %f ms\n", name.c_str(), elapsedTime);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return elapsedTime;
	}

	template<typename T>
	static inline void memcpy(T* dst, T* src, int num, cudaMemcpyKind kind = cudaMemcpyHostToDevice, int offset = 0)
	{
		checkCudaErrors(cudaMemcpy(dst + offset, src, num * sizeof(T), kind));
	}

	template<typename T>
	static inline void memset(T* mem, int nn) {
		checkCudaErrors(cudaMemset(mem, 0, nn * sizeof(T)));
	}

	template<typename T>
	static inline void malloc(T*& mem, int nn) {
		checkCudaErrors(cudaMalloc((void**)&mem, nn * sizeof(T)));
	}

	template<typename T>
	static inline void getValue(T* ret, T* mem, int N) {
		checkCudaErrors(cudaMemcpy(ret, mem + N, sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	static inline T getValue(T* mem, int N) {
		T ret;
		checkCudaErrors(cudaMemcpy(&ret, mem + N, sizeof(T), cudaMemcpyDeviceToHost));
		return ret;
	}

    static void* cubTempStorage;
	static size_t cubTempStorageBytes;

	static void allocCubTemp();
	template<typename T>
	static void sortRegist(int* label, int* label_out, int num_items);

	template<typename T>
	static void sort(T* d_in, T* d_out, int* label, int* label_out, int num_items);
	
	template<typename T>
	static void minReduceRegist(int num_items);

	template<typename T>
	static void maxReduceRegist(int num_items);

	template<typename T>
	static void minReduce(T* d_in, T* d_out, int num_items);

	template<typename T>
	static void maxReduce(T* d_in, T* d_out, int num_items);

	template<typename T>
	static float maxReduce(const std::string& name, T* d_in, T* d_out, int num_items);

	template<typename T>
	static float minReduce(const std::string& name, T* d_in, T* d_out, int num_items);
};
