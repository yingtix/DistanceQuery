#include "DistanceQuery.h"

#include "MyCuda/MainCodeDistanceQuery.cuh"
#include "MyCuda/MainCodeBuildBVH.cuh"
#include "MyCuda/Type.cuh"
#include "MyCuda/TriDist.cuh"
#include "MyCuda/TempData.cuh"
#include "ObjLoader.h"
#include "MyCuda/utils.cuh"

#define MAX_TIME 1000

g_transf initTransA;
g_transf initTransB;
float thetaA = 0.01;
float thetaB = 0.01;
float3 axisA = make_float3(1, 1, 1);
float3 axisB = make_float3(-1, 1, 1);
float3 offsetB = make_float3(0, 0, 0);
int count = 0;
float sum = 0;
float tmax = 0;
std::vector<DistanceResult> results;
DistanceResult result;

void DistanceQueryApp::init(int benchmarkId, bool calculateMinD, std::string path)
{
    this->calculateMinD = calculateMinD;
    this->benchmarkId = benchmarkId;
    std::string filename1 = path;
    std::string filename2 = path;
    float scale1 = 1;
    float scale2 = 1;
    switch (benchmarkId)
    {
    case 1:
        filename1 += "tools_1000000_A.obj";
        filename2 += "tools_1000000_A.obj";
        break;
    case 2:
        filename1 += "ring1.obj";
        filename2 += "ring1.obj";
        break;
    case 3:
        filename1 += "voronoimodel3.obj";
        filename2 += "voronoimodel3.obj";
        break;
    case 4:
        filename1 += "voronoimodel3.obj";
        filename2 += "voronoimodel3.obj";
        break;
    case 5:
        filename1 += "tools_1000000_A.obj";
        filename2 += "rosetta_1000000_B.obj";
        break;
    case 6:
        filename1 += "voronoisphere.obj";
        filename2 += "voronoisphere.obj";
        scale1 = 4;
        scale2 = 2;
        break;
    case 7:
        filename1 += "tools_1000000_A.obj";
        filename2 += "tools_1000000_B_transform_benchmark.obj";
        break;
    case 8:
        break;
    case 9:
        filename1 += "soup_1000000_A.obj";
        filename2 += "soup_1000000_B.obj";
        break;
    case 10:
        filename1 += "intersection_1000000_A.obj";
        filename2 += "intersection_1000000_B.obj";
        break;
    }
    readobjfile(filename1, numTriA, this->vtxsA, calculateMinD, scale1);
    readobjfile(filename2, numTriB, this->vtxsB, calculateMinD, scale2);

    if (numTriA > numTriB)
    {
        auto temp = this->vtxsA; this->vtxsA = this->vtxsB; this->vtxsB = temp;
        auto temp1 = numTriA; numTriA = numTriB; numTriB = temp1;
    }

    gpu_data = new TempData();
    gpu_data->use<g_box>("bvhA_nodes");
    gpu_data->use<g_box>("bvhB_nodes");
    gpu_data->use<float3>("bvhA_vtxs", 3);
    gpu_data->use<float3>("bvhB_vtxs", 3);
    gpu_data->use<float3>("bvhA_vtxsUpdate", 3);
    gpu_data->useLarge<float>("outDis", (1 << 25));
    gpu_data->useLarge<int>("outId1", (1 << 25));
    gpu_data->useLarge<int>("outId2", (1 << 25));
    gpu_data->use<int>("mutex");
    gpu_data->useLarge<int>("bvtt1_a", (1 << 25)); 
    gpu_data->useLarge<int>("bvtt1_b", (1 << 25));
    gpu_data->useLarge<float>("bvtt1_min", (1 << 25));
    gpu_data->useLarge<int>("bvtt2_a", (1 << 25));
    gpu_data->useLarge<int>("bvtt2_b", (1 << 25));
    gpu_data->useLarge<float>("bvtt2_min", (1 << 25));

    g_transf initTrans;
    initTrans._off = { 0, 0, 0 };
    initTrans._trf = make_float3x3(1.f);

    BuildBVH::MainProcess(numTriA, vtxsA, deepA, gpu_data, true, calculateMinD, sort_vtxA);

    initTrans._off = { 0, 0, 0 };
    initTrans._trf = make_float3x3(1.f);
    BuildBVH::MainProcess(numTriB, vtxsB, deepB, gpu_data, false, calculateMinD, sort_vtxB);
    printf("%d %d\n", numTriA, numTriB);
    printf("%d %d\n", deepA, deepB);

    if (calculateMinD) {
        utils::minReduceRegist<float>(1 << 25);
    }
    else {
        utils::maxReduceRegist<float>(1 << 25);
    }
    utils::allocCubTemp();

    const int tt = MAX_TIME;
    initTransA._off = { 0, 0, 0 };
    //initTrans._off = { 0, 0, 0 };
    initTransA._trf = make_float3x3(1.f);

    axisA = normalize(axisA);
    axisB = normalize(axisB);
    switch (benchmarkId)
    {
    case 1:
        initTransB._off = make_float3(0, 100, -150);
        thetaA = 0;
        thetaB = 0;
        offsetB = make_float3(0, -1, 0);
        break;
    case 2:
        initTransB._off = make_float3(-75, 0, 0);
        break;
    case 3:
        initTransB._off = make_float3(-150, 0, 0);
        offsetB = make_float3(-0.5, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    case 4:
        initTransB._off = make_float3(-1500, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    case 5:
        initTransB._off = make_float3(-10, 0, 0);
        thetaA = 0;
        
        break;
    case 6:
        initTransB._off = make_float3(0, 2, 0);
        axisA = make_float3(0, 1, 0);
        axisB = make_float3(0, 1, 0);
        thetaA = 0.03490658503988659153847381536977;
        thetaB = 0.03490658503988659153847381536977;
        break;
    case 7:
        initTransB._off = make_float3(0, 0, 0);
        thetaA = 0;
        thetaB = 0;
        offsetB = make_float3(0, 0, 0.25);
        break;
    case 8:
        axisA = make_float3(0, 0, 1);
        axisB = make_float3(0, 0, 1);
        break;
    case 9:
        initTransB._off = make_float3(0, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    case 10:
        initTransB._off = make_float3(-1500, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    }
    initTransB._trf = make_float3x3(1.f);
}

void DistanceQueryApp::run()
{
    const int tt = MAX_TIME;
    g_transf initTransA;
    g_transf initTransB;
    initTransA._off = { 0, 0, 0 };
    //initTrans._off = { 0, 0, 0 };
    initTransA._trf = make_float3x3(1.f);

    float thetaA = 0.01;
    float thetaB = 0.01;
    float3 axisA = make_float3(1, 1, 1);
    float3 axisB = make_float3(-1, 1, 1);
    axisA = normalize(axisA);
    axisB = normalize(axisB);

    float3 offsetB = make_float3(0, 0, 0);

    switch (benchmarkId)
    {
    case 1:
        //initTransB._off = make_float3(0, 100, -150);
        initTransB._off = make_float3(0, 0, 0);
        thetaA = 0;
        thetaB = 0;
        offsetB = make_float3(0, -1, 0);
        break;
    case 2:
        //initTransB._off = make_float3(-75, 0, 0);
        initTransB._off = make_float3(0, 0, 0);
        break;
    case 3:
        //initTransB._off = make_float3(-150, 0, 0);
        initTransB._off = make_float3(0, 0, 0);
        offsetB = make_float3(-1, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    case 4:
        /*initTransB._off = make_float3(-1500, 0, 0);*/
        initTransB._off = make_float3(0, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    case 5:
        /*initTransB._off = make_float3(-15, 0, 0);*/
        initTransB._off = make_float3(0, 0, 0);
        break;
    case 6:
        /*initTransB._off = make_float3(0, 2, 0);*/
        initTransB._off = make_float3(0, 0, 0);
        axisA = make_float3(0, 1, 0);
        axisB = make_float3(0, 1, 0);
        break;
    case 7:
        /*initTransB._off = make_float3(0, 0, 0);*/
        initTransB._off = make_float3(0, 0, 0);
        thetaA = 0;
        thetaB = 0;
        offsetB = make_float3(0, 0, 0.25);
        break;
    case 8:
        axisA = make_float3(0, 0, 1);
        axisB = make_float3(0, 0, 1);
        break;
    case 9:
        initTransB._off = make_float3(0, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    case 10:
        initTransB._off = make_float3(-1500, 0, 0);
        thetaA = 0;
        thetaB = 0;
        break;
    }
    initTransB._trf = make_float3x3(1.f);


    //initTrans = initTrans.inverse();

    float sum = 0;
    float max = 0;
    std::vector<DistanceResult> results;
    DistanceResult result;
    result.id1 = 0;
    result.id2 = 0;
    for (int i = 0; i < tt; i++)
    {
        result.lab = i;
        initTransA._trf = initTransA._trf * rotation(axisA, thetaA);

        initTransB._trf = initTransB._trf * rotation(axisB, thetaB);

        initTransB._off += offsetB;

        g_transf initTrans;
        float3x3 temp = getTrans(initTransB._trf);
        initTrans._trf = temp * initTransA._trf;
        initTrans._off = initTransA._off - initTransB._off;
        initTrans._off = temp * initTrans._off;
        //nitTrans._off -= make_float3(0, 0, 0.25f);
        cudaDeviceSynchronize();
        DistanceQuery::MainProcessAllTime(numTriA, numTriB, deepA, deepB, initTrans, 1000, gpu_data, calculateMinD, result);
        //printf("%f\n", result.sum_time);
        printf("%d: %f %f\n", i, result.sum_time, result.min);
        max = fmax(max, result.sum_time);
        sum += result.sum_time;
        results.push_back(result);
    }
    float avg = sum / tt;
    printf("avg %f; max %f\n", sum / tt, max);
    int d_sum = 0, d_cnt = 0;
    int maxlength = 0;
    int lengthCnt[11];
    memset(lengthCnt, 0, 11 * sizeof(int));
    for (const auto& result : results)
    {
        //if (result.sum_time > avg * 1.2)
        //{
        //    printf("%d %f\n", result.lab, result.sum_time);
        //}
        for (int deep : result.proDeeps)
        {
            lengthCnt[deep]++;
            d_sum += deep;
            d_cnt++;
        }
        for (int length : result.bvtts)
        {
            if (length > maxlength)
            {
                maxlength = length;
            }


        }
    }
    printf("avg deep %f\n", 1.f * d_sum / d_cnt);
    printf("max length %d\n", maxlength);
    for (int i = 1; i < 10; i++)
    {
        printf("length %d: %d\n", i, lengthCnt[i]);
    }
}

void DistanceQueryApp::step() 
{
    //initTrans = initTrans.inverse();

    result.id1 = 0;
    result.id2 = 0;

    result.lab = count;
    initTransA._trf = initTransA._trf * rotation(axisA, thetaA);

    initTransB._trf = initTransB._trf * rotation(axisB, thetaB);

    initTransB._off += offsetB;

    g_transf initTrans;
    float3x3 temp = getTrans(initTransB._trf);
    initTrans._trf = temp * initTransA._trf;
    initTrans._off = initTransA._off - initTransB._off;
    initTrans._off = temp * initTrans._off;
    //nitTrans._off -= make_float3(0, 0, 0.25f);
    cudaDeviceSynchronize();
    DistanceQuery::MainProcessAllTime(numTriA, numTriB, deepA, deepB, initTrans, 1000, gpu_data, calculateMinD, result);
    //printf("%f\n", result.sum_time);
    printf("%d: %f %f\n", count, result.sum_time, result.min);
    tmax = fmax(tmax, result.sum_time);
    float3 pt1, pt2;
    float disNew = triDist(initTransA.apply(sort_vtxA[result.id1 * 3]), initTransA.apply(sort_vtxA[result.id1 * 3 + 1]), initTransA.apply(sort_vtxA[result.id1 * 3 + 2])
        ,   initTransB.apply(sort_vtxB[result.id2 * 3]), initTransB.apply(sort_vtxB[result.id2 * 3 + 1]), initTransB.apply(sort_vtxB[result.id2 * 3 + 2]), pt1, pt2);
    
    ptA[0] = pt1.x; ptA[1] = pt1.y; ptA[2] = pt1.z;
    ptB[0] = pt2.x; ptB[1] = pt2.y; ptB[2] = pt2.z;

    float disOld = triDist(initTransA.apply(sort_vtxA[result.oldId1 * 3]), initTransA.apply(sort_vtxA[result.oldId1 * 3 + 1]), initTransA.apply(sort_vtxA[result.oldId1 * 3 + 2])
        , initTransB.apply(sort_vtxB[result.oldId2 * 3]), initTransB.apply(sort_vtxB[result.oldId2 * 3 + 1]), initTransB.apply(sort_vtxB[result.oldId2 * 3 + 2]), pt1, pt2);
    
    if (disOld < disNew + 1e-10)
    {
        ptA[0] = pt1.x; ptA[1] = pt1.y; ptA[2] = pt1.z;
        ptB[0] = pt2.x; ptB[1] = pt2.y; ptB[2] = pt2.z;
    }
    else {
        result.oldId1 = result.id1;
        result.oldId2 = result.id2;
    }
    //float3 AA = initTransA.apply(sort_vtxA[result.id1 * 3]);
    //ptA[0] = sort_vtxA[result.id1 * 3].x;
    //ptA[1] = sort_vtxA[result.id1 * 3].y;
    //ptA[2] = sort_vtxA[result.id1 * 3].z;
    //
    //float3 BB = initTransB.apply(sort_vtxB[result.id2 * 3]);
    //ptB[0] = BB.x;
    //ptB[1] = BB.y;
    //ptB[2] = BB.z;

    sum += result.sum_time;
    results.push_back(result);
    count += 1;
}

void DistanceQueryApp::end()
{
    delete gpu_data;
    delete[] vtxsA;
    delete[] vtxsB;
}

void DistanceQueryApp::getResultTriID(int& id0, int& id1) 
{
    id0 = result.id1;
    id1 = result.id2;
}
