#pragma once
#pragma once
//#include <cuda.h>

#include "cuda_runtime.h"
//#include "cudart_platform.h"
//#include "device_launch_parameters.h"
#include <string>
#include <iostream>

class Timer
{
	float alltime;
	cudaEvent_t start, stop;
public:
	Timer()
	{
		alltime = 0;
	}
	inline void tick()
	{
		float elapsedTime = 0.0;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}
	inline void tock(const std::string& name)
	{
		float elapsedTime;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		std::cout << name << " time: " << elapsedTime << " ms\n";
		alltime += elapsedTime;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
};
