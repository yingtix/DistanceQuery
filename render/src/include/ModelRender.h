#ifndef OBJ_DRAW_MODELRENDER_H
#define OBJ_DRAW_MODELRENDER_H

#include <filesystem>

#include <glm/glm.hpp>

#include "Camera.h"
#include "Model.h"
#include "Shader.h"

class ModelRender {
public:
    void init(Model* model, const std::filesystem::path& vertPath, const std::filesystem::path& fragPath);

    void init(Model* model, Shader shader);

    void draw(const Camera& camera, const glm::mat4& appMMat, bool drawLine);

    Shader& getShader();

    Model* getModel();

    glm::u32vec2 getVBO();

private:
    void initOpenGLObject();

private:
    Model* m_model{nullptr};
    Shader   m_shader{};
    uint32_t VAO{};
    uint32_t VBO[2]{};
    uint32_t EBO{};

    glm::mat4 mMat{1};
    glm::mat4 vMat{1};
    glm::mat4 mvMat{1};
    glm::mat4 projMat{1};
    glm::mat4 invTrMat{1};
};


#endif //OBJ_DRAW_MODELRENDER_H
