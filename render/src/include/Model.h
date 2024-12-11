#ifndef OBJ_DRAW_MODEL_H
#define OBJ_DRAW_MODEL_H

#include <cstdint>

#include <filesystem>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "ModelMatrix.h"

class Model {
public:
    uint32_t                                   m_sizeof_T{4}; // sizeof(float) == 4
    uint32_t                                   m_num_vtx{0}; // 顶点数量
    uint32_t                                   m_num_tri{0}; // 三角形数量
    std::shared_ptr<std::vector<glm::vec3>>    m_vtx{nullptr};
    std::shared_ptr<std::vector<glm::u32vec3>> m_tri{nullptr};
    std::shared_ptr<std::vector<glm::vec3>>    m_nrm{nullptr}; // 顶点的法向量，由相邻面的法向量插值得到
    ModelMatrix                                mMat{};

public:
    Model();

    bool initModel(const std::filesystem::path& model_path, double scale, glm::vec3 shift, bool swap_xyz);

    void memFree();

private:
};

#endif //OBJ_DRAW_MODEL_H
