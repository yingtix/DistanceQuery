struct float3;
class TempData;
struct g_transf;

namespace BuildBVH {
	void MainProcess(int numtri, float3* vtxs, int& ret_deep, TempData* gpu_data, bool isA, bool isMin, float3*& sort_vtx);
}
