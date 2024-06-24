#pragma once

#include <string>
#include "vulkan/vulkan.h"

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

	VkDescriptorSet getDescriptorSet() const { return m_descriptorSet; }

	void resize(uint32_t width, uint32_t height);

	uint32_t getWidth() const { return m_width; }
	uint32_t getHeight() const { return m_height; }

	VkImage getHandle() const { return m_image; }

private:
	void allocateMemory(uint64_t size);
	void release();

private:
	uint32_t m_width = 0, m_height = 0;

	VkDescriptorSet m_descriptorSet = nullptr;

	VkImage m_image = nullptr;
	VkImageView m_imageView = nullptr;
	VkDeviceMemory m_memory = nullptr;
	VkSampler m_sampler = nullptr;

	ImageType m_type = ImageType::None;

	VkBuffer m_buffer = nullptr;
	VkDeviceMemory m_bufferMemory = nullptr;

	size_t m_size = 0;

	std::string m_filepath;
};