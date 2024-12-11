#version 430 core

layout(location = 0) in vec3 position;

out vec4 fragColor;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
uniform vec3 line_color;

void main(void)
{
    fragColor = vec4(line_color, 1.0);
}