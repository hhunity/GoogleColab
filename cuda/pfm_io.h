%%writefile pfm_io.h
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cstring>

// PFMフォーマット用のシンプルな構造体（グレースケール: channels=1, RGB: channels=3）
struct PFMImage {
    int width = 0;
    int height = 0;
    int channels = 0;            // 1 or 3
    std::vector<float> data;     // size = width * height * channels, row-major
};

inline float byteswap_float(float v) {
    uint32_t x;
    static_assert(sizeof(float) == sizeof(uint32_t), "float size unexpected");
    std::memcpy(&x, &v, sizeof(float));
    x = ((x & 0x000000FFu) << 24) |
        ((x & 0x0000FF00u) << 8)  |
        ((x & 0x00FF0000u) >> 8)  |
        ((x & 0xFF000000u) >> 24);
    std::memcpy(&v, &x, sizeof(float));
    return v;
}

inline bool read_pfm(const std::string& path, PFMImage& img) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::fprintf(stderr, "Failed to open %s\n", path.c_str());
        return false;
    }
    std::string magic;
    ifs >> magic;
    if (magic != "PF" && magic != "Pf") {
        std::fprintf(stderr, "Unsupported PFM magic (need PF or Pf)\n");
        return false;
    }
    img.channels = (magic == "PF") ? 3 : 1;

    // ヘッダー読み取り（コメント行スキップ）
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
    std::string scaletok = next_token();
    if (wtok.empty() || htok.empty() || scaletok.empty()) {
        std::fprintf(stderr, "Malformed PFM header\n");
        return false;
    }
    img.width = std::stoi(wtok);
    img.height = std::stoi(htok);
    float scale = std::stof(scaletok);
    bool little_endian = (scale < 0.0f);  // PFM規約: 符号がエンディアン
    float scale_abs = std::abs(scale);

    // データ読み取り
    ifs.get(); // consume single whitespace after header
    size_t n = static_cast<size_t>(img.width) * img.height * img.channels;
    img.data.resize(n);
    ifs.read(reinterpret_cast<char*>(img.data.data()), n * sizeof(float));
    if (!ifs) {
        std::fprintf(stderr, "Failed to read PFM pixel data\n");
        return false;
    }

    // 必要ならバイトスワップ
    const bool host_little = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
    if (little_endian != host_little) {
        for (size_t i = 0; i < n; ++i) {
            img.data[i] = byteswap_float(img.data[i]);
        }
    }
    // スケールを適用
    if (scale_abs != 0.0f && scale_abs != 1.0f) {
        for (size_t i = 0; i < n; ++i) {
            img.data[i] *= scale_abs;
        }
    }
    // PFMは下から上に格納される仕様なので、行を反転してトップダウンに戻す
    size_t row_elems = static_cast<size_t>(img.width) * img.channels;
    for (int y = 0; y < img.height / 2; ++y) {
        float* row_top = img.data.data() + static_cast<size_t>(y) * row_elems;
        float* row_bottom = img.data.data() + static_cast<size_t>(img.height - 1 - y) * row_elems;
        for (size_t i = 0; i < row_elems; ++i) {
            std::swap(row_top[i], row_bottom[i]);
        }
    }
    return true;
}

inline bool write_pfm(const std::string& path, const PFMImage& img) {
    if (img.channels != 1 && img.channels != 3) {
        std::fprintf(stderr, "PFM write: channels must be 1 or 3\n");
        return false;
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "Failed to open %s for write\n", path.c_str());
        return false;
    }
    // PFMは符号でエンディアンを示す。ホストがリトルエンディアンなら負のスケールを書く。
    const bool host_little = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
    float scale = host_little ? -1.0f : 1.0f;
    ofs << (img.channels == 3 ? "PF\n" : "Pf\n");
    ofs << img.width << " " << img.height << "\n";
    ofs << scale << "\n";
    // PFMは下から上に走査する仕様なので、行を反転して書き出す
    size_t row_elems = static_cast<size_t>(img.width) * img.channels;
    for (int y = img.height - 1; y >= 0; --y) {
        const float* row_ptr = img.data.data() + static_cast<size_t>(y) * row_elems;
        ofs.write(reinterpret_cast<const char*>(row_ptr), row_elems * sizeof(float));
    }
    return static_cast<bool>(ofs);
}
