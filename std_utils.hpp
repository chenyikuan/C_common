#ifndef STD_UTILS_HPP_
#define STD_UTILS_HPP_

#include <dirent.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>


inline void strToLower(std::string& str)
{
    int i = 0;
    while (str[i])
    {
        str[i] = (std::tolower(str[i]));
        i++;
    }
}


inline std::vector<std::string> getAllFiles(const std::string& path_, std::string subfix = "")
{
    std::string path = path_;
    if (path[path.length()-1] != '/') {
        path += "/";
    }
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        std::cout << "Error folder path: " << path << std::endl;
        exit(-1);
    }
    strToLower(subfix);
    if (subfix[0] != '.') {
        subfix = "." + subfix;
    }
    // printf("subfix: %s\n", subfix.c_str());
    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == 8) {
            std::string fn(ptr->d_name);
            std::string fn_subfix(ptr->d_name+fn.length()-subfix.length());
            strToLower(fn_subfix);
            if (subfix == "." || subfix == "*" || strcmp(fn_subfix.c_str(), subfix.c_str()) == 0)
                files.push_back(path + ptr->d_name);
        } else if (ptr->d_type == 10) {
            continue;
        } else if (ptr->d_type == 4) {
            /// folder
        }
    }
    closedir(dir);
    return files;
}


inline std::vector<std::string> stringSplit(const std::string& str, char delim) {
    // std::stringstream ss(str);
    // std::string item;
    // std::vector<std::string> elems;
    // while (std::getline(ss, item, delim)) {
    //     if (!item.empty()) {
    //         elems.push_back(item);
    //     }
    // }
    // return elems;

    std::vector<std::string> elems;
    int i = 0;
    for (int j = 0; j < str.length() - 1; j++) {
        if (str[j] == delim) {
            elems.push_back(str.substr(i, j-i));
            i = j+1;
        }
    }
    if (i < str.length()) {
        elems.push_back(str.substr(i, str.length()-i));
    }
    return elems;
}


#endif