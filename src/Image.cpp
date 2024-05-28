#include "Image.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"

#include "Application.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace Utils
{
    static uint32_t bytesPerPixel(const ImageType type)
    {
        switch (type)
        {
        case ImageType::RGBA:
            return 4;
        case ImageType::RGBA32F:
            return 16;
        }
        return 0;
    }
}

Image::Image(const std::string_view path)
    : m_path(path)
{
    int width, height, channels;
    uint8_t* data = nullptr;

    if (stbi_is_hdr(m_path.c_str()))
    {
        data = reinterpret_cast<uint8_t*>(stbi_loadf(m_path.c_str(), &width, &height, &channels, STBI_rgb_alpha));
        m_type = ImageType::RGBA32F;
    }
    else
    {
        data = stbi_load(m_path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        m_type = ImageType::RGBA;
    }

    if (!data)
    {
        std::cerr << "Failed to load image: " << m_path << std::endl;
        return;
    }

    m_width = width;
    m_height = height;

    allocateMemory(static_cast<uint64_t>(m_width) * m_height * Utils::bytesPerPixel(m_type));
    setData(data);
    stbi_image_free(data);
}

Image::Image(const uint32_t width, const uint32_t height, const ImageType type, const void* data)
    : m_width(width), m_height(height), m_type(type)
{
    allocateMemory(static_cast<uint64_t>(m_width) * m_height * Utils::bytesPerPixel(m_type));
    if (data)
        setData(data);
}

Image::~Image()
{
    release();
}

void Image::allocateMemory(const uint64_t size)
{
    m_size = size;
    m_data = malloc(m_size);
}

void Image::release()
{
    if (m_data)
    {
        free(m_data);
        m_data = nullptr;
    }
}

void Image::setData(const void* data) const
{
    memcpy(m_data, data, m_size);
}

void Image::resize(const uint32_t width, const uint32_t height)
{
    m_width = width;
    m_height = height;
    m_size = static_cast<uint64_t>(m_width) * m_height * Utils::bytesPerPixel(m_type);
    m_data = realloc(m_data, m_size);
}
