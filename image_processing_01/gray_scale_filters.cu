#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>

#pragma pack(push, 1)
struct BmpFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BmpInfoHeader {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

bool loadBmp(const char* path, int& width, int& height, std::vector<uint8_t>& outPixels) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "Failed to open input file '%s'\n", path);
        return false;
    }

    BmpFileHeader fileHeader;
    BmpInfoHeader infoHeader;

    if (std::fread(&fileHeader, sizeof(fileHeader), 1, fp) != 1) {
        std::fprintf(stderr, "Failed to read BMP file header from '%s'\n", path);
        std::fclose(fp);
        return false;
    }

    if (std::fread(&infoHeader, sizeof(infoHeader), 1, fp) != 1) {
        std::fprintf(stderr, "Failed to read BMP info header from '%s'\n", path);
        std::fclose(fp);
        return false;
    }

    if (fileHeader.bfType != 0x4D42) {
        std::fprintf(stderr, "'%s' is not a valid BMP file\n", path);
        std::fclose(fp);
        return false;
    }

    if (infoHeader.biPlanes != 1 || infoHeader.biCompression != 0 || infoHeader.biBitCount != 24) {
        std::fprintf(stderr, "Unsupported BMP format in '%s'. Only 24-bit uncompressed BMP is supported.\n", path);
        std::fclose(fp);
        return false;
    }

    width = infoHeader.biWidth;
    int absHeight = std::abs(infoHeader.biHeight);
    bool topDown = infoHeader.biHeight < 0;

    int rowSize = ((width * 3 + 3) / 4) * 4;
    int dataSize = rowSize * absHeight;
    std::vector<uint8_t> raw(dataSize);

    if (std::fseek(fp, fileHeader.bfOffBits, SEEK_SET) != 0) {
        std::fprintf(stderr, "Failed to seek BMP pixel data in '%s'\n", path);
        std::fclose(fp);
        return false;
    }

    if (std::fread(raw.data(), 1, dataSize, fp) != static_cast<size_t>(dataSize)) {
        std::fprintf(stderr, "Failed to read BMP pixel data from '%s'\n", path);
        std::fclose(fp);
        return false;
    }

    std::fclose(fp);

    outPixels.assign(width * absHeight * 3, 0);

    for (int row = 0; row < absHeight; ++row) {
        int destRow = topDown ? row : (absHeight - 1 - row);
        const uint8_t* src = raw.data() + row * rowSize;
        uint8_t* dst = outPixels.data() + destRow * width * 3;

        for (int x = 0; x < width; ++x) {
            dst[3 * x + 0] = src[3 * x + 2];
            dst[3 * x + 1] = src[3 * x + 1];
            dst[3 * x + 2] = src[3 * x + 0];
        }
    }

    height = absHeight;
    return true;
}

bool saveBmp(const char* path, int width, int height, const std::vector<uint8_t>& pixels) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) {
        std::fprintf(stderr, "Failed to open output file '%s'\n", path);
        return false;
    }

    int rowSize = ((width * 3 + 3) / 4) * 4;
    int dataSize = rowSize * height;

    BmpFileHeader fileHeader;
    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = static_cast<uint32_t>(54 + dataSize);
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = 54;

    BmpInfoHeader infoHeader;
    infoHeader.biSize = 40;
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = static_cast<uint32_t>(dataSize);
    infoHeader.biXPelsPerMeter = 2835;
    infoHeader.biYPelsPerMeter = 2835;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    if (std::fwrite(&fileHeader, sizeof(fileHeader), 1, fp) != 1 ||
        std::fwrite(&infoHeader, sizeof(infoHeader), 1, fp) != 1) {
        std::fprintf(stderr, "Failed to write BMP header to '%s'\n", path);
        std::fclose(fp);
        return false;
    }

    std::vector<uint8_t> row(rowSize, 0);
    for (int y = 0; y < height; ++y) {
        const uint8_t* src = pixels.data() + y * width * 3;
        for (int x = 0; x < width; ++x) {
            row[3 * x + 0] = src[3 * x + 2];
            row[3 * x + 1] = src[3 * x + 1];
            row[3 * x + 2] = src[3 * x + 0];
        }
        if (std::fwrite(row.data(), 1, rowSize, fp) != static_cast<size_t>(rowSize)) {
            std::fprintf(stderr, "Failed to write BMP pixel data to '%s'\n", path);
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

void applyGrayCPU(std::vector<uint8_t>& pixels) {
    size_t pixelCount = pixels.size() / 3;
    for (size_t i = 0; i < pixelCount; ++i) {
        uint8_t r = pixels[3 * i + 0];
        uint8_t g = pixels[3 * i + 1];
        uint8_t b = pixels[3 * i + 2];
        uint8_t gray = static_cast<uint8_t>(std::round(0.299f * r + 0.587f * g + 0.114f * b));
        pixels[3 * i + 0] = gray;
        pixels[3 * i + 1] = gray;
        pixels[3 * i + 2] = gray;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s input.bmp output.bmp\n", argv[0]);
        return 1;
    }

    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixelsCPU;

    if (!loadBmp(inputPath, width, height, pixelsCPU)) {
        return 1;
    }

    applyGrayCPU(pixelsCPU);

    if (!saveBmp(outputPath, width, height, pixelsCPU)) {
        return 1;
    }

    std::printf("Converted '%s' to grayscale and saved as '%s'\n", inputPath, outputPath);
    return 0;
}
