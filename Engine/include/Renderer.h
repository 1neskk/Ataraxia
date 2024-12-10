#pragma once

#include <cstdint>
#include <memory>

#include "Scene.h"
#include "Image.h"
#include "Camera.h"

class Renderer
{
	struct HitRecord
	{
		float t;
		glm::vec3 worldPos;
		glm::vec3 worldNormal;
		Sphere sphere;
		uint32_t id;
	};

public:
    Renderer();
    ~Renderer();
    
    void onResize(uint32_t width, uint32_t height);
    void Render(Camera& camera, const Scene& scene);

    [[nodiscard]] std::shared_ptr<Image> getImage() const { return m_image; }

	Settings& getSettings() { return m_settings; }
    void resetFrameIndex() { m_frameIndex = 1; }

	static __device__ HitRecord traceRay(const Ray& ray, const Sphere* spheres, size_t numSpheres);
    static __device__ HitRecord rayMiss(const Ray& ray);
    static __device__ HitRecord rayHit(const Ray& ray, float tmin, int index, const Sphere* spheres);
	static __device__ glm::vec4 perPixel(uint32_t x, uint32_t y, uint32_t width, const Sphere* spheres,
        size_t numSpheres, const DeviceCamera& d_camera, const Material* materials, size_t numMaterials, uint32_t frameIndex,
        const Light* lights, size_t numLights);
private:
	void allocateDeviceMemory(const Scene& scene);
	void freeDeviceMemory();

private:
    const Scene* m_scene = nullptr;
	Sphere* d_spheres_ = nullptr; // device spheres
	Material* d_materials_ = nullptr; // device materials
	Light* d_lights_ = nullptr; // device lights

    glm::vec4* d_accumulation_ = nullptr; // device accumulation buffer

	std::shared_ptr<Image> m_image = nullptr;
    uint32_t* h_imageData_ = nullptr;  // host image data
    uint32_t* d_imageData_ = nullptr; // device image data

	Settings m_settings;
    uint32_t m_frameIndex = 1;

    uint32_t m_width = 0, m_height = 0;
};

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres,
    size_t numSpheres, const DeviceCamera d_camera, const Material* materials, size_t numMaterials, glm::vec4* accumulation,
    uint32_t frameIndex, const Light* lights, size_t numLights);

namespace colorUtils
{
    __device__ inline uint32_t vec4ToRGBA(const glm::vec4& color)
    {
        const auto r = static_cast<uint8_t>(color.x * 255.0f);
        const auto g = static_cast<uint8_t>(color.y * 255.0f);
        const auto b = static_cast<uint8_t>(color.z * 255.0f);
        const auto a = static_cast<uint8_t>(color.w * 255.0f);

        return (a << 24) | (b << 16) | (g << 8) | r;
    }

    __device__ inline uint32_t vec3ToRGBA(const glm::vec3& color)
    {
        const auto r = static_cast<uint8_t>(color.x * 255.0f);
        const auto g = static_cast<uint8_t>(color.y * 255.0f);
        const auto b = static_cast<uint8_t>(color.z * 255.0f);

        return (0xFF << 24) | (b << 16) | (g << 8) | r;
    }
}
