#include "ModelRender.h"

#include <glad/glad.h>

void ModelRender::init(Model* model, const std::filesystem::path& vertPath, const std::filesystem::path& fragPath) {
    m_model  = model;
    m_shader = Shader{vertPath, fragPath};
    initOpenGLObject();
}

void ModelRender::init(Model* model, Shader shader) {
    m_model  = model;
    m_shader = shader;
    initOpenGLObject();
}

void ModelRender::draw(const Camera& camera, const glm::mat4& appMMat, bool drawLine) {
    m_shader.use();
    mMat     = appMMat;
    mMat *= m_model->mMat.getModelMatrix();
    vMat     = camera.getViewMatrix();
    projMat  = camera.getProjMatrix();
    mvMat    = vMat * mMat;
    invTrMat = transpose(inverse(mvMat));

    m_shader.setMat4Uniform("mv_matrix", value_ptr(mvMat));
    m_shader.setMat4Uniform("proj_matrix", value_ptr(projMat));
    m_shader.setMat4Uniform("norm_matrix", value_ptr(invTrMat));

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    if (drawLine) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDisable(GL_CULL_FACE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_CULL_FACE);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glDrawElements(GL_TRIANGLES, m_model->m_num_tri * 3, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

Shader& ModelRender::getShader() {
    return m_shader;
}

Model* ModelRender::getModel() {
    return m_model;
}

void ModelRender::initOpenGLObject() {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(2, VBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, m_model->m_num_vtx * 3 * m_model->m_sizeof_T, m_model->m_vtx->data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, m_model->m_num_vtx * 3 * m_model->m_sizeof_T, m_model->m_nrm->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_model->m_num_tri * 3 * m_model->m_sizeof_T,
                 m_model->m_tri->data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

glm::u32vec2 ModelRender::getVBO() {
    return {VBO[0], VBO[1]};
}
