#pragma once

#include <iostream>
#include <string>
#include <cstdint>

enum class ImageType
{
    None = 0,
    RGBA,
    RGBA32F
};

class Image
{
public:
    Image(std::string_view path);
    Image(uint32_t width, uint32_t height, ImageType type, const void* data = nullptr);
    ~Image();

    void setData(const void* data);

    void resize(uint32_t width, uint32_t height);

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    ImageType getType() const { return m_type; }


private:
    void allocateMemory(uint64_t size);
    void release();

private:
    uint32_t m_width = 0, m_height = 0;
    ImageType m_type = ImageType::None;
    size_t m_size = 0;
    std::string m_path;
    void* m_data = nullptr;
};