%%writefile pgm_io.h
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> data;
};

inline bool read_pgm(const std::string& path, Image& img) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::fprintf(stderr, "Failed to open %s\n", path.c_str());
        return false;
    }
    std::string magic;
    ifs >> magic;
    if (magic != "P5") {
        std::fprintf(stderr, "Unsupported PGM format (need P5)\n");
        return false;
    }
    auto next_token = [&]() -> std::string {
        std::string tok;
        while (ifs >> tok) {
            if (!tok.empty() && tok[0] == '#') {
                std::string dummy;
                std::getline(ifs, dummy);
                continue;
            }
            return tok;
        }
        return {};
    };
    std::string wtok = next_token();
    std::string htok = next_token();
    std::string maxtok = next_token();
    if (wtok.empty() || htok.empty() || maxtok.empty()) {
        std::fprintf(stderr, "Malformed PGM header\n");
        return false;
    }
    img.width = std::stoi(wtok);
    img.height = std::stoi(htok);
    int maxv = std::stoi(maxtok);
    if (maxv != 255) {
        std::fprintf(stderr, "Only 8-bit PGM (max 255) supported\n");
        return false;
    }
    ifs.get(); // consume single whitespace after header
    img.data.resize(static_cast<size_t>(img.width) * img.height);
    ifs.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    if (!ifs) {
        std::fprintf(stderr, "Failed to read pixel data\n");
        return false;
    }
    return true;
}

inline bool write_pgm(const std::string& path, const Image& img) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "Failed to open %s for write\n", path.c_str());
        return false;
    }
    ofs << "P5\n" << img.width << " " << img.height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return static_cast<bool>(ofs);
}
