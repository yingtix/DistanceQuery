#include "GLProgram.h"
class render
{
private:
    int benchmarkId;
    GLProgram program;
    Model model0;
    Model model1;
    std::string filePath;

public:
    void init(int benchmarkId, bool calculateMinD);
    void setPath(std::string filePath);
    void run();
};