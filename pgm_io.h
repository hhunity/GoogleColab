#pragma once
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> data;
};

bool read_pgm(const std::string& path, Image& img);
bool write_pgm(const std::string& path, const Image& img);
