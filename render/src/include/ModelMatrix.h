#ifndef OBJ_DRAW_MODELMATRIX_H
#define OBJ_DRAW_MODELMATRIX_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

class ModelMatrix {
private:
    glm::mat4 translationMatrix{1.0f};
    glm::mat4 rotationMatrix{1.0f};
    glm::mat4 scalingMatrix{1.0f};
    glm::mat4 modelMatrix{1.0f};

public:
    void translate(const glm::vec3& translation) {
        translationMatrix = glm::translate(glm::mat4(1.0f), translation);
        updateModelMatrix();
    }

    void rotate(float angle, const glm::vec3& axis) {
        rotationMatrix = glm::rotate(glm::mat4(1.0f), angle, axis);
        updateModelMatrix();
    }

    void rotate(const glm::quat& r_quat) {
        rotationMatrix = toMat4(r_quat);
        updateModelMatrix();
    }

    void scale(const glm::vec3& scale) {
        scalingMatrix = glm::scale(glm::mat4(1.0f), scale);
        updateModelMatrix();
    }

    glm::mat4 getModelMatrix() const {
        return modelMatrix;
    }

private:
    void updateModelMatrix() {
        modelMatrix = translationMatrix * rotationMatrix * scalingMatrix;
    }
};

#endif //OBJ_DRAW_MODELMATRIX_H
