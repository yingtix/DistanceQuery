#pragma once
#include <string>
//#include "Render/Render.h"
//#include "MyCuda/CudaGLInterp.h"
class Application
{
public:
	virtual void init(int benckmarkId, bool calculateMinD, std::string path) = 0;
	virtual void run() = 0;
	virtual void step() = 0;
	virtual void end() = 0;
};