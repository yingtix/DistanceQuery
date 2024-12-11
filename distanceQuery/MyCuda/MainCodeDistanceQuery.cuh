#include "device_launch_parameters.h"
#include <vector>
struct g_transf;
struct DistanceResult;
class TempData;

namespace DistanceQuery {
	void MainProcessAllTime(int numTriA, int numTriB, int deepA, int deepB, const g_transf rigids, const float initDis, TempData* data, bool isMin, DistanceResult&);
	void MainProcess(int numTriA, int numTriB, int deepA, int deepB, const g_transf rigids, const float initDis, TempData* data, bool isMin, DistanceResult&);
}