#ifndef OBJ_DRAW_GLPROGRAM_H
#define OBJ_DRAW_GLPROGRAM_H
#define SI static inline
#define SC static constexpr

#include <iostream>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Camera.h"
#include "ModelRender.h"
#include "Shader.h"

#include "App/DistanceQuery.h"
class GLProgram
{
public:
    struct Light
    {
        float ambient[4];
        float diffuse[4];
        float specular[4];
        float position[3];
    };

    struct Material
    {
        float ambient[4];
        float diffuse[4];
        float specular[4];
        float shininess;
    };

    
public:
    static void init(Model *model0, Model *model1, glm::vec3 points[], double rotateAngles[], glm::vec3 rotateAxis[], glm::vec3 translates[], glm::vec3 offset[], int benchmarkId, bool calculateMinD);

    static void run();

    static void cleanup();

    static void initOpenGl();

    static void initImGui();

    static void setCallbackFunc();

    static void initData(Model *model0, Model *model1);

    static void initLinePoints(glm::vec3 points[]);

    static void drawModel();

    static void drawImgui();

    static void installLights(Shader &shader, int key);

    static void updateLine(glm::vec3 point1, glm::vec3 point2);

    static void toggle();

    static void record();

    static void capPNG();
    SI std::string filePath;

private:
    SI GLFWwindow* window{ nullptr };
    SI DistanceQueryApp distanceApp{};
    SI int windowSize[2]{1024, 1024};
    SI std::string windowTitle{"distance query"};
    SI glm::vec4 clearColor{0.85f, 0.85f, 0.85f, 1.0f};

    SI glm::mat4 appMMat{1};
    SI glm::mat4 lastMMat{1};
    SI glm::mat4 addedMMat{1};
    SI glm::mat4 rotateMMat{1};

    // data for dynamic models
    SI double angle0{ 0 };
    SI double angle1{ 0 };
    SI double rotateAngle0{ 0 };
    SI double rotateAngle1{ 0 };
    SI glm::vec3 rotateAxis0{ 0, 1, 0 };
    SI glm::vec3 rotateAxis1{ 0, 1, 0 };
    SI glm::vec3 offset0{ 0, 0, 0 };
    SI glm::vec3 offset1{ 0, 0, 0 };
    SI glm::vec3 translate0{ 0, 0, 0 };
    SI glm::vec3 translate1{ 0, 0, 0 };

    SI bool drawLine{false};
    SI bool drawConnectLine{true};
    SI float sceneScale{0.05f};
    SC float sceneScaleMin{0.0002f};
    SC float sceneScaleMax{50.0f};
    SI float rotateAngleX{0.0f};
    SI float rotateAngleY{0.0f};
    SI double rotateAngle{0};
    SI bool show_model0{true};
    SI bool show_model1{true};

    SI glm::vec3 line_points[2]{};
    SI glm::vec3 line_color{1, 0.5, 0};
    SI float line_width{3.0f};
    SI Shader line_shader{};
    SI uint32_t line_vao{};
    SI uint32_t line_vbo{};

private:
    SI ModelRender mr0{};
    SI ModelRender mr1{};

    SI glm::vec3 defaultCamPos{0, 0, 3};
    SI Camera camera{defaultCamPos, {0, 0, 0}};

    SI Light light{
        {0.5f, 0.5f, 0.5f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f},
        {10.0f, 10.0f, 10.0f}};

    SI Material materia11{
       {0.0f, 0.6f, 0.0f, 1.0f},
       {0.0f, 0.3f, 0.0f, 1.0f},
       {0.0f, 0.1f, 0.0f, 1.0f},
       20.0f };

    SI Material material2{
        {0.0f, 0.0f, 0.6f, 1.0f},
        {0.0f, 0.0f, 0.3f, 1.0f},
        {0.0f, 0.0f, 0.1f, 1.0f},
        20.0f};
    
    SI float globalAmbient[4]{0.7f, 0.7f, 0.7f, 1.0f};

    SI bool leftButtonPress{false};
    SI bool middleButtonPress{false};
    SI bool middleButtonRepeat{false};
    SI bool mousePosInit{false};

    SI double lastXPos{0};
    SI double lastYPos{0};

    SI bool showImGuiWindows{true};
    SI float fpsUpdateInterval{0.5f};
    SI float framerate{0.0f};
    SI double runtime{0};
    SI bool doAnimation{ false };
    SI bool recordVideo{ false };

private:
    SI bool glInit{false};
    SI bool imGuiInit{false};
};

#undef SI
#undef SC

#endif // OBJ_DRAW_GLPROGRAM_H
