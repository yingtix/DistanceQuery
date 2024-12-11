#include "GLProgram.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <sstream>
#include <iomanip>
#include <glm/gtx/io.hpp>
#include "stb_image.h"
#include "stb_image_write.h"
#include "TextLoader.h"

void GLProgram::init(Model* model0, Model* model1, glm::vec3 points[], double rotateAngles[], glm::vec3 rotateAxis[], glm::vec3 translates[], glm::vec3 offset[], int benchmarkID, bool calculateMinD)
{
    initOpenGl();
    initImGui();
    setCallbackFunc();
    initData(model0, model1);
    initLinePoints(points);
    distanceApp.init(benchmarkID, calculateMinD, filePath);
    rotateAngle0 = rotateAngles[0];
    rotateAngle1 = rotateAngles[1];
    rotateAxis0 = rotateAxis[0];
    rotateAxis1 = rotateAxis[1];
    translate0 = translates[0];
    translate1 = translates[1];
    offset0 = offset[0];
    offset1 = offset[1];
}

void GLProgram::run()
{
    while (!glfwWindowShouldClose(window))
    {
        if (doAnimation) 
        {
            distanceApp.step();
            
            updateLine(glm::vec3(distanceApp.ptA[0], distanceApp.ptA[1], distanceApp.ptA[2]), glm::vec3(distanceApp.ptB[0], distanceApp.ptB[1], distanceApp.ptB[2]));
            drawModel();
            drawImgui();
            glfwPollEvents();
            glfwSwapBuffers(window);
        }
        else 
        {
            drawModel();
            drawImgui();
            glfwPollEvents();
            glfwSwapBuffers(window);
        }
    }
}

void GLProgram::initOpenGl()
{
    if (glInit)
        return;

    if (!glfwInit())
        return;
    glfwWindowHint(GLFW_SAMPLES, 4);
    window = glfwCreateWindow(windowSize[0], windowSize[1], windowTitle.c_str(), nullptr, nullptr);
    if (!window)
        return;
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        return;

    //==================================================================================================================

    // glfwSwapInterval(1); // 开启垂直同步
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glfwGetFramebufferSize(window, &windowSize[0], &windowSize[1]);
    glViewport(0, 0, windowSize[0], windowSize[1]);
    const auto aspect = static_cast<float>(windowSize[0]) / static_cast<float>(windowSize[1]);
    camera.setAspect(aspect);

    glInit = true;
}

void GLProgram::initImGui()
{
    if (imGuiInit)
        return;
    const std::filesystem::path text_path{R"(.\render\assets\text\ui-utf8.txt)"};
    TextLoader::loadText(text_path);
    constexpr char glsl_version[] = "#version 430";
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    float xScale{1.f}, yScale{1.f};
    GLFWmonitor *primaryMonitor = glfwGetPrimaryMonitor();
    glfwGetMonitorContentScale(primaryMonitor, &xScale, &yScale);
    io.Fonts->AddFontFromFileTTF(R"(./render/assets/fonts/font1.ttf)", 20 * yScale, nullptr,
                                 io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
    imGuiInit = true;
}

void GLProgram::setCallbackFunc()
{
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height)
                                   {
        glViewport(0, 0, width, height);
        windowSize[0]     = width;
        windowSize[1]     = height;
        const auto aspect = static_cast<float>(width) / static_cast<float>(height);
        camera.setAspect(aspect); });

    glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods)
                       {
        if (ImGui::GetIO().WantCaptureKeyboard) {
            ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
        } else {
            if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
        }
        if (!ImGui::GetIO().WantTextInput) {
            if (key == GLFW_KEY_L && action == GLFW_PRESS) { drawLine = !drawLine; }
            if (key == GLFW_KEY_V && action == GLFW_PRESS) { showImGuiWindows = !showImGuiWindows; }
        }
        // 不管ImGui是否需要处理事件，都执行
        {
        } });

    glfwSetScrollCallback(window, [](GLFWwindow *window, double xOffset, double yOffset)
                          {
        if (ImGui::GetIO().WantCaptureMouse) {
            ImGui_ImplGlfw_ScrollCallback(window, xOffset, yOffset);
        } else {
            sceneScale = glm::clamp(sceneScale + static_cast<float>(yOffset * 2.0f * sceneScale / sceneScaleMax),
                                    sceneScaleMin, sceneScaleMax);
        } });

    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods)
                               {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        if (!ImGui::GetIO().WantCaptureMouse) {
            switch (button) {
                case GLFW_MOUSE_BUTTON_LEFT: {
                    if (action == GLFW_PRESS) {
                        leftButtonPress = true;
                    } else if (action == GLFW_RELEASE) {
                        leftButtonPress = false;

                        lastMMat = addedMMat * rotateMMat * lastMMat;

                        rotateAngle  = 0;
                        rotateAngleX = 0;
                        rotateAngleY = 0;
                    }
                    break;
                }
                case GLFW_MOUSE_BUTTON_MIDDLE: {
                    if (action == GLFW_PRESS) {
                        middleButtonPress  = true;
                        middleButtonRepeat = false;
                    } else if (action == GLFW_RELEASE) {
                        middleButtonPress = false;
                        if (!middleButtonRepeat) {
                            camera.setPosition(defaultCamPos);
                        }
                    }
                }
                default:
                    break;
            }
        } });

    glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos)
                             {
        ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

        if (!mousePosInit) {
            lastXPos     = xpos;
            lastYPos     = ypos;
            mousePosInit = true;
        }
        auto deltaX = xpos - lastXPos;
        auto deltaY = ypos - lastYPos;

        lastXPos = xpos;
        lastYPos = ypos;

        if (leftButtonPress) {
            rotateAngleY += static_cast<float>(deltaX * 0.005);
            rotateAngleX += static_cast<float>(deltaY * 0.005);
        }
        if (middleButtonPress) {
            middleButtonRepeat = true;
            auto pos = camera.getPosition();
            pos.x -= static_cast<float>(deltaX * 0.005);
            pos.y += static_cast<float>(deltaY * 0.005);
            camera.setPosition(pos);
        } });
}

void GLProgram::initData(Model *model0, Model *model1)
{
    const std::filesystem::path shader_path[2]{
        R"(.\render\assets\shaders\VertexShader.glsl)",
        R"(.\render\assets\shaders\FragmentShader.glsl)"};

    mr0.init(model0, shader_path[0], shader_path[1]);
    mr1.init(model1, shader_path[0], shader_path[1]);
    // mr1.init(model1, mr0.getShader());
}

void GLProgram::initLinePoints(glm::vec3 *points)
{
    line_points[0] = points[0];
    line_points[1] = points[1];
    const std::filesystem::path line_shader_path[2]{
        R"(.\render\assets\shaders\VertexShader_Line.glsl)",
        R"(.\render\assets\shaders\FragmentShader_Line.glsl)"};
    line_shader = {line_shader_path[0], line_shader_path[1]};
    glGenVertexArrays(1, &line_vao);
    glBindVertexArray(line_vao);
    glGenBuffers(1, &line_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(line_points), line_points, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GLProgram::updateLine(glm::vec3 point1, glm::vec3 point2) 
{
    line_points[0] = point1;
    line_points[1] = point2;
}

void GLProgram::toggle()
{
    doAnimation = !doAnimation;
    if (!recordVideo)
    {
        recordVideo = true;
        std::cout << "Start recording video" << std::endl;
    }
    else
    {
        recordVideo = false;
        std::cout << "Stop recording video" << std::endl;

        // Convert images to video
        system("ffmpeg -r 60 -f image2 -s 1920x1080 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p 1.mp4");
        system("ffmpeg -i 1.mp4 -vf vflip video.gif");
    }
}

void GLProgram::record()
{
    
}

void GLProgram::capPNG()
{
    static int frameNumber = 0;
    if (recordVideo)
    {
        printf("record p\n");
        // Capture screen
        GLubyte* pixels = new GLubyte[3 * windowSize[0] * windowSize[1]];
        glReadPixels(0, 0, windowSize[0], windowSize[1], GL_RGB, GL_UNSIGNED_BYTE, pixels);

        // Save to file
        std::ostringstream filename;
        filename << "frame" << std::setfill('0') << std::setw(4) << frameNumber++ << ".png";
        stbi_write_png(filename.str().c_str(), windowSize[0], windowSize[1], 3, pixels, windowSize[1] * 3);

        delete[] pixels;
    }
}

void GLProgram::drawModel()
{
    glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    appMMat = glm::mat4{1};
    addedMMat = rotate(glm::mat4{1.0f}, rotateAngleX, glm::vec3{1, 0, 0}) * rotate(glm::mat4{1.0f}, rotateAngleY, glm::vec3{0, 1, 0});
    rotateMMat = rotate(glm::mat4{1.0f}, static_cast<float>(rotateAngle), glm::vec3{0, 1, 0});
    appMMat *= scale(glm::mat4{1}, {sceneScale, sceneScale, sceneScale});
    appMMat *= addedMMat;
    appMMat *= rotateMMat;
    appMMat *= lastMMat;

    if (doAnimation) 
    {
        offset0 += translate0;
        offset1 += translate1;
        angle0 += rotateAngle0;
        angle1 += rotateAngle1;
    }


    if (show_model0)
    {
        installLights(mr0.getShader(), 0);
        mr0.getModel()->mMat.translate(offset0);
        mr0.getModel()->mMat.rotate(angle0, rotateAxis0);
        mr0.draw(camera, appMMat , drawLine);
    }
    if (show_model1)
    {
        installLights(mr1.getShader(), 1);
        mr1.getModel()->mMat.translate(offset1);
        mr1.getModel()->mMat.rotate(angle1, rotateAxis1);
        mr1.draw(camera, appMMat, drawLine);
    }

    if (drawConnectLine)
    {
        line_shader.use();
        auto mMat = appMMat;
        auto vMat = camera.getViewMatrix();
        auto projMat = camera.getProjMatrix();
        auto mvMat = vMat * mMat;
        line_shader.setMat4Uniform("mv_matrix", value_ptr(mvMat));
        line_shader.setMat4Uniform("proj_matrix", value_ptr(projMat));
        line_shader.setVec3Uniform("line_color", value_ptr(line_color));

        glGenVertexArrays(1, &line_vao);
        glBindVertexArray(line_vao);
        glGenBuffers(1, &line_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(line_points), line_points, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glBindVertexArray(line_vao);
        glBindBuffer(GL_ARRAY_BUFFER, line_vbo);

        glLineWidth(line_width);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINES, 0, sizeof(line_points) / sizeof(glm::vec3));
        glLineWidth(1.0f);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
}

void GLProgram::drawImgui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (showImGuiWindows)
    {
        if (glfwGetTime() - runtime > fpsUpdateInterval)
        {
            runtime = glfwGetTime();
            framerate = ImGui::GetIO().Framerate;
        }
        ImGui::Begin("Console(V)", &showImGuiWindows);
        ImGui::ColorEdit3("clear color", &clearColor.x,
                          ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayRGB |
                              ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_PickerHueWheel);
        ImGui::SliderFloat(TEXT(refresh_interval), &fpsUpdateInterval, .05f, 2.f, nullptr,
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::Text("%.1f FPS  ( %.3f ms/frame )", framerate, 1000.0f / framerate);

        ImGui::SliderFloat(TEXT(sceneScale), &sceneScale, sceneScaleMin, sceneScaleMax, nullptr,
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);

        ImGui::Checkbox(TEXT(draw_line), &drawLine);
        //ImGui::Button();
        if (ImGui::Button("Toggle Animation")) 
        {
            toggle();
        }

        
        if (ImGui::Button("Record Video")) 
        {
            record();
        }
        capPNG();

        ImGui::Checkbox(TEXT(show_model0), &show_model0);
        ImGui::SameLine();
        ImGui::Text(" (vtx: %d tri: %d)", mr0.getModel()->m_num_vtx, mr0.getModel()->m_num_tri);
        ImGui::Checkbox(TEXT(show_model1), &show_model1);
        ImGui::SameLine();
        ImGui::Text(" (vtx: %d tri: %d)", mr1.getModel()->m_num_vtx, mr1.getModel()->m_num_tri);

        ImGui::Checkbox(TEXT(draw_connect_line), &drawConnectLine);

        if (drawConnectLine)
        {
            ImGui::SliderFloat(TEXT(line_width), &line_width, 1.0f, 5.0f, nullptr, ImGuiSliderFlags_AlwaysClamp);
            ImGui::ColorEdit3(TEXT(line_color), glm::value_ptr(line_color),
                              ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayRGB |
                                  ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_PickerHueWheel);
        }
        ImGui::End();
    }
    
    //ImGui::Text("The min distance is %f", distanceApp.minDist);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLProgram::installLights(Shader &shader, int key)
{
    if (key == 0) {
        shader.use();
        shader.setVec4Uniform("globalAmbient", globalAmbient);
        shader.setVec4Uniform("light.ambient", light.ambient);
        shader.setVec4Uniform("light.diffuse", light.diffuse);
        shader.setVec4Uniform("light.specular", light.specular);
        shader.setVec3Uniform("light.position", light.position);
        shader.setVec4Uniform("material.ambient", materia11.ambient);
        shader.setVec4Uniform("material.diffuse", materia11.diffuse);
        shader.setVec4Uniform("material.specular", materia11.specular);
        shader.setFloatUniform("material.shininess", materia11.shininess);
    }
    else {
        shader.use();
        shader.setVec4Uniform("globalAmbient", globalAmbient);
        shader.setVec4Uniform("light.ambient", light.ambient);
        shader.setVec4Uniform("light.diffuse", light.diffuse);
        shader.setVec4Uniform("light.specular", light.specular);
        shader.setVec3Uniform("light.position", light.position);
        shader.setVec4Uniform("material.ambient", material2.ambient);
        shader.setVec4Uniform("material.diffuse", material2.diffuse);
        shader.setVec4Uniform("material.specular", material2.specular);
        shader.setFloatUniform("material.shininess", material2.shininess);
    }
}

void GLProgram::cleanup()
{
    if (!glInit || !imGuiInit)
        return;
    glInit = false;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
