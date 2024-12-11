#ifndef OBJ_DRAW_SHADER_H
#define OBJ_DRAW_SHADER_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <glm/glm.hpp>

class Shader
{
public:
    uint32_t ID;

public:
    Shader() = default;
    Shader(const std::filesystem::path& vertexPath, const std::filesystem::path& fragmentPath);
    void use() const;
    void setFloatUniform(const std::string& name, float value) const;
    void setVec3Uniform(const std::string& name, const float* value) const;
    void setVec4Uniform(const std::string& name, const float* value) const;
    void setMat4Uniform(const std::string& name, const float* value) const;

private:
    void checkCompileErrors(uint32_t shader, std::string type);
};


#endif //OBJ_DRAW_SHADER_H
