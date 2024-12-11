#include "GLProgram.h"
#include "render.h"

const int benchmarkId = 7;
const bool calculateMinD = true;
const std::string path = "D:/code/DistanceQuery/data/";

int main()
{
	render app;
	app.setPath(path);
	app.init(benchmarkId, calculateMinD);
	app.run();
}