#include "render.h"
const glm::vec3 axisA(1.0f, 1.0f, 1.0f);
const glm::vec3 axisB(-1.0f, 1.0f, 1.0f);
void render::setPath(std::string path)
{
    this->filePath = path;
}

void render::init(int benchmarkId, bool calculateMinD)
{
    program.filePath = this->filePath;
    switch (benchmarkId)
    {
    case 1:
    {   
        std::string path = this->filePath + "tools_1000000_A.obj";
        model0.initModel(path, 1, glm::vec3(0, 0, 0), false);
        model1.initModel(path, 1, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = {0.0f, 0.0f};
        glm::vec3 rotateAxis[2] = {axisA, axisB};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(0, 100, -150)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.f, 0.0f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }

    case 2:
    {   
        model0.initModel(this->filePath + "ring1.obj", 1, glm::vec3(0, 0, 0), false);
        model1.initModel(this->filePath + "ring1.obj", 1, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = {0.01, 0.01};
        glm::vec3 rotateAxis[2] = {axisA, axisB};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(-75, 0, 0)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.f, 0.0f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }

    case 3:
    {
        // init compelete
        model0.initModel(this->filePath + "voronoimodel3.obj", 1, glm::vec3(0, 0, 0), false);
        model1.initModel(this->filePath + "voronoimodel3.obj", 1, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = {0.f, 0.f};
        glm::vec3 rotateAxis[2] = {axisA, axisB};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(-150, 0, 0)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-0.5f, 0.f, 0.0f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }

    case 4:
    {
        // init compelete
        model0.initModel(this->filePath + "voronoimodel3.obj", 1, glm::vec3(0, 0, 0), false);
        model1.initModel(this->filePath + "voronoimodel3.obj", 1, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = {0.f, 0.f};
        glm::vec3 rotateAxis[2] = {axisA, axisB};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(-1500, 0, 0)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.f, 0.0f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }

    case 5:
    {
        // init compelete
        model0.initModel(this->filePath + "tools_1000000_A.obj", 1, glm::vec3(0, 0, 0), false);
        model1.initModel(this->filePath + "rosetta_1000000_B.obj", 1, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = {0.f, 0.01f};
        glm::vec3 rotateAxis[2] = {axisA, axisB};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(-10, 0, 0)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.f, 0.0f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }

    case 6:
    {
        model0.initModel(this->filePath + "voronoisphere.obj", 4, glm::vec3(0, 0, 0), false);
        model1.initModel(this->filePath + "voronoisphere.obj", 2, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = { 0.03490658503988659153847381536977, 0.03490658503988659153847381536977};
        glm::vec3 rotateAxis[2] = {glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(0, 2, 0)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.f, 0.0f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }

    case 7:
    {
        model0.initModel(this->filePath + "tools_1000000_A.obj", 1, glm::vec3(0, 0, 0), false);
        model1.initModel(this->filePath + "tools_1000000_B_transform_benchmark.obj", 1, glm::vec3(0, 0, 0), false);
        glm::vec3 points[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
        double rotateAngle[2] = {0.0f, 0.0f};
        glm::vec3 rotateAxis[2] = {glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)};
        glm::vec3 offset[2] = {glm::vec3(0.f, 0.f, 0.f), glm::vec3(0, 0, 0)};
        glm::vec3 translate[2] = {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.f, 0.25f)};
        program.init(&model0, &model1, points, rotateAngle, rotateAxis, translate, offset, benchmarkId, calculateMinD);
        break;
    }
    }
}

void render::run()
{
    program.run();
}