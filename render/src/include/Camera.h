#ifndef OBJ_DRAW_CAMERA_H
#define OBJ_DRAW_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera();

    explicit Camera(glm::vec3 position, glm::vec3 center, glm::vec3 worldUp = {0, 1, 0});

    explicit Camera(glm::vec3 position, float pitch, float yaw, glm::vec3 worldUp = {0, 1, 0});

    virtual ~Camera() = default;

protected:
    void updateViewMatrix();

    void updateProjMatrix();

public:
    enum class ProjMode : int {
        ORTHO       = 1,
        PERSPECTIVE = 2
    };

protected:
    glm::vec3 position;
    glm::vec3 forward{};
    glm::vec3 right{};
    glm::vec3 up{};
    glm::vec3 worldUp;
    float     pitch{};
    float     yaw;

protected:
    float fovy;
    float aspect;
    float zNear;
    float zFar;

protected:
    glm::mat4 viewMatrix;
    glm::mat4 perspectiveMatrix;
    glm::mat4 orthoMatrix;

public:
    ProjMode projMode{ProjMode::PERSPECTIVE};

protected:
    inline static constexpr float PI{3.141592653589793f};
    inline static constexpr float pitch_limit{0.001f};

public:
    [[nodiscard]] const glm::vec3& getPosition() const;

    [[nodiscard]] const glm::vec3& getForward() const;

    [[nodiscard]] const glm::vec3& getRight() const;

    [[nodiscard]] const glm::vec3& getUp() const;

    [[nodiscard]] const glm::vec3& getWorldUp() const;

    [[nodiscard]] float getPitch() const;

    [[nodiscard]] float getYaw() const;

    [[nodiscard]] float getFovy() const;

    [[nodiscard]] float getZNear() const;

    [[nodiscard]] float getZFar() const;

    [[nodiscard]] const glm::mat4& getViewMatrix() const;

    [[nodiscard]] const glm::mat4& getProjMatrix() const;

    [[nodiscard]] float getAspect() const;

    void setAspect(float newAspect);

    void setProjMod(ProjMode projMod);

    void setPosition(const glm::vec3& position);

protected:
    void updateCamera();
};


#endif //OBJ_DRAW_CAMERA_H
