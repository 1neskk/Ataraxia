#include "Image.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"

#define WIN32_LEAN_AND_MEAN
#ifdef _WIN32
#include <Windows.h>
#endif

#include <GL/gl.h>

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
		std::cerr << "Failed to load image: " << m_path << " \n";
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
    if (m_data)
    {
        release();
    }

    m_size = size;
	m_data.reset(malloc(m_size));

	if (!m_data)
	{
		std::cerr << "Failed to allocate memory for Image: " << m_path << " \n";
	}
	std::cout << "Allocated memory: " << m_size << " bytes\n";
}

void Image::release()
{
    if (m_data)
    {
        m_data.reset();
		std::cout << "Released memory: " << m_size << " bytes\n";
    }
}

void Image::setData(const void* data) const
{
    memcpy(m_data.get(), data, m_size);
}

void Image::resize(const uint32_t width, const uint32_t height)
{
	if (m_width == width && m_height == height)
		return;

    m_width = width;
    m_height = height;
    m_size = static_cast<uint64_t>(m_width) * m_height * Utils::bytesPerPixel(m_type);

    if (m_data)
    {
		std::cout << "Resizing memory: " << m_size << " bytes\n";
		m_data.reset(realloc(m_data.release(), m_size));
		if (!m_data)
		{
			std::cerr << "Failed to reallocate memory for Image resize: " << m_path << " \n";
		}
    }
    else
    {
		allocateMemory(m_size);
    }
}

uint32_t Image::getTextureID() const 
{
    uint32_t textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_data.get());

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}
