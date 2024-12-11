//#include "CudaGLInterp.h"
//#include "../Render/Render.h"
//#include "Dynamic/Dynamic.h"
//
//CudaGLInterp::CudaGLInterp(Dynamic* _d, Render* _r): dyn(_d), render(_r){}
//
//void CudaGLInterp::linkBuffer() {
//	cudaGraphicsGLRegisterBuffer(&cudaVBO1, render->VBOs[0], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaVBO2, render->VBOs[1], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaVBO3, render->VBOs[2], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaVBO4, render->VBOs[3], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaEBO1, render->EBOs[0], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaEBO2, render->EBOs[1], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaEBO3, render->EBOs[2], cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsGLRegisterBuffer(&cudaEBO4, render->EBOs[3], cudaGraphicsMapFlagsWriteDiscard);
//}
//void CudaGLInterp::map() {
//	cudaGraphicsMapResources(1, &cudaVBO1, 0);
//	cudaGraphicsMapResources(1, &cudaVBO2, 0);
//	cudaGraphicsMapResources(1, &cudaEBO1, 0);
//	cudaGraphicsMapResources(1, &cudaEBO2, 0);
//	cudaGraphicsMapResources(1, &cudaVBO3, 0);
//	cudaGraphicsMapResources(1, &cudaVBO4, 0);
//	cudaGraphicsMapResources(1, &cudaEBO3, 0);
//	cudaGraphicsMapResources(1, &cudaEBO4, 0);
//	size_t num_bytes;
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_cloth_vert, &num_bytes, cudaVBO1);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_cloth_nrm, &num_bytes, cudaVBO2);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_obstacle_vert, &num_bytes, cudaVBO3);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_obstacle_nrm, &num_bytes, cudaVBO4);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_cloth_face, &num_bytes, cudaEBO1);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_cloth_edge, &num_bytes, cudaEBO2);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_obstacle_face, &num_bytes, cudaEBO3);
//	cudaGraphicsResourceGetMappedPointer((void**)&dyn->render_obstacle_edge, &num_bytes, cudaEBO4);
//}
//void CudaGLInterp::unmap() {
//	cudaGraphicsUnmapResources(1, &cudaVBO1, 0);
//	cudaGraphicsUnmapResources(1, &cudaVBO2, 0);
//	cudaGraphicsUnmapResources(1, &cudaEBO1, 0);
//	cudaGraphicsUnmapResources(1, &cudaEBO2, 0);
//	cudaGraphicsUnmapResources(1, &cudaVBO3, 0);
//	cudaGraphicsUnmapResources(1, &cudaVBO4, 0);
//	cudaGraphicsUnmapResources(1, &cudaEBO3, 0);
//	cudaGraphicsUnmapResources(1, &cudaEBO4, 0);
//}
//
//void CudaGLInterp::clear() {
//	cudaGraphicsUnregisterResource(cudaVBO1);
//	cudaGraphicsUnregisterResource(cudaVBO2);
//	cudaGraphicsUnregisterResource(cudaEBO1);
//	cudaGraphicsUnregisterResource(cudaEBO2);
//	cudaGraphicsUnregisterResource(cudaVBO3);
//	cudaGraphicsUnregisterResource(cudaVBO4);
//	cudaGraphicsUnregisterResource(cudaEBO3);
//	cudaGraphicsUnregisterResource(cudaEBO4);
//}