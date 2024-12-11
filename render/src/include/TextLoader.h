#ifndef OBJ_DRAW_TEXTLOADER_H
#define OBJ_DRAW_TEXTLOADER_H

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>

class TextLoader {
public:
    static void loadText(const std::filesystem::path& text_path) {
        if (!exists(text_path)) {
            printf("error path: %s ---> line %d\n", __FILE__, __LINE__);
            return;
        }

        std::ifstream fs{text_path};

        if (!fs.is_open()) {
            printf("error file open: %s ---> line %d\n", __FILE__, __LINE__);
            return;
        }

        std::string            line{};
        std::string            key{}, value{};
        std::string::size_type pos{};
        while (std::getline(fs, line)) {
            pos = line.find_first_of(':');
            if (pos == std::string::npos) continue;
            key = line.substr(0, pos);
            if (pos == line.size() - 1) {
                value = nullptr;
            } else {
                value = line.substr(pos + 1, line.size());
            }
            text_map.insert({key, value});
        }
    }

    static std::string_view getValue(const std::string& key) {
        const auto it = text_map.find(key);
        if (it == text_map.end()) {
            return text_map.at("null");
        }
        return text_map.at(key);
    }

private:
    static inline std::map<std::string, std::string> text_map{{"null", ""}};
};

#define TEXT(x) (TextLoader::getValue(#x).data())


#endif //OBJ_DRAW_TEXTLOADER_H
