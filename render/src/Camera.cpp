#include "Camera.h"

Camera::Camera() : Camera({0, 0, 0}, {0, 0, -1}, {0, 1, 0}) {}

Camera::Camera(glm::vec3 position, glm::vec3 center, glm::vec3 worldUp) :
        position{position}, worldUp{worldUp},
        zNear{.1f}, zFar{10000.f}, fovy{1.0472f}, aspect{1.f},
        viewMatrix{}, perspectiveMatrix{}, orthoMatrix{} {
    forward = glm::normalize(center - position);
    right   = glm::normalize(glm::cross(forward, worldUp));
    up      = glm::normalize(glm::cross(right, forward));
    pitch   = glm::acos(this->forward.y);

    if (forward.x < 0)
        yaw = PI - glm::asin(this->forward.z / glm::sin(pitch));
    else
        yaw = glm::asin(this->forward.z / glm::sin(pitch));
    if (yaw < 0) yaw += 2 * PI;

    updateViewMatrix();
    updateProjMatrix();
}

Camera::Camera(glm::vec3 position, float pitch, float yaw, glm::vec3 worldUp) :
        pitch{pitch}, yaw{yaw},
        position{position}, worldUp{worldUp},
        zNear{.1f}, zFar{10000.f}, fovy{1.0472f}, aspect{1.f},
        viewMatrix{}, perspectiveMatrix{}, orthoMatrix{} {
    pitch   = glm::clamp(pitch, pitch_limit, PI - pitch_limit);
    forward = glm::vec3{
            glm::sin(pitch) * glm::cos(yaw),
            glm::cos(pitch),
            glm::sin(pitch) * glm::sin(yaw)
    };
    right   = glm::normalize(glm::cross(forward, worldUp));
    up      = glm::normalize(glm::cross(right, forward));

    updateViewMatrix();
    updateProjMatrix();
}

void Camera::updateCamera() {
    forward = glm::vec3{
            glm::sin(pitch) * glm::cos(yaw),
            glm::cos(pitch),
            glm::sin(pitch) * glm::sin(yaw)
    };
    right   = glm::normalize(glm::cross(forward, worldUp));
    up      = glm::normalize(glm::cross(right, forward));
    updateViewMatrix();
}

const glm::mat4& Camera::getViewMatrix() const {
    return viewMatrix;
}

const glm::mat4& Camera::getProjMatrix() const {
    if (projMode == ProjMode::PERSPECTIVE)
        return perspectiveMatrix;
    return orthoMatrix;
}

float Camera::getAspect() const {
    return aspect;
}

void Camera::setAspect(float newAspect) {
    aspect = newAspect;
    updateProjMatrix();
}

void Camera::updateViewMatrix() {
    viewMatrix = glm::lookAt(position, position + forward, worldUp);
}

void Camera::updateProjMatrix() {
    perspectiveMatrix = glm::perspective(fovy, aspect, zNear, zFar);
    float tanHalfFovy      = glm::tan(fovy / 2.f);
    float distanceToOrigin = glm::distance(position, glm::vec3{0});
    float k                = tanHalfFovy * distanceToOrigin;
    orthoMatrix = glm::ortho(-aspect * k, aspect * k, -k, k, zNear, zFar);
}

const glm::vec3& Camera::getPosition() const {
    return position;
}

const glm::vec3& Camera::getForward() const {
    return forward;
}

const glm::vec3& Camera::getRight() const {
    return right;
}

const glm::vec3& Camera::getUp() const {
    return up;
}

const glm::vec3& Camera::getWorldUp() const {
    return worldUp;
}

float Camera::getPitch() const {
    return pitch;
}

float Camera::getYaw() const {
    return yaw;
}

float Camera::getFovy() const {
    return fovy;
}

float Camera::getZNear() const {
    return zNear;
}

float Camera::getZFar() const {
    return zFar;
}

void Camera::setProjMod(Camera::ProjMode newProjMod) {
    Camera::projMode = newProjMod;
}

void Camera::setPosition(const glm::vec3& newPosition) {
    Camera::position = newPosition;
    updateViewMatrix();
}
