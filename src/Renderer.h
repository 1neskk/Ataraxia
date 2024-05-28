#pragma once

#include <cstdint>
#include <memory>

#include "Scene.h"
#include "Application.h"
#include "Image.h"
#include "vec/Vec4.h"

class Renderer
{
public:
    Renderer();
    ~Renderer();
    
    void onResize(uint32_t width, uint32_t height);

    void Render(const Scene& scene);

    static __device__ Vec4 perPixel(uint32_t x, uint32_t y, uint32_t width,
                                    uint32_t height, const Sphere* spheres, size_t numSpheres);

    std::shared_ptr<Image> getImage() const { return m_image; }

    static __device__ bool intersect(const Ray& ray, const Sphere* spheres, size_t numSpheres, float& t);

private:
    const Scene* m_scene = nullptr;

    std::shared_ptr<Image> m_image;
    uint32_t* h_imageData_ = nullptr;
    uint32_t* d_imageData_ = nullptr;

    uint32_t m_width = 0, m_height = 0;

};

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres, size_t numSpheres);

namespace colorUtils
{
    __device__ inline uint32_t vec4ToRGBA(const Vec4& color)
    {
        const auto r = static_cast<uint8_t>(color.x * 255.0f);
        const auto g = static_cast<uint8_t>(color.y * 255.0f);
        const auto b = static_cast<uint8_t>(color.z * 255.0f);
        const auto a = static_cast<uint8_t>(color.w * 255.0f);

        return (a << 24) | (b << 16) | (g << 8) | r;
    }

    __device__ inline uint32_t vec3ToRGBA(const Vec3& color)
    {
        const auto r = static_cast<uint8_t>(color.x * 255.0f);
        const auto g = static_cast<uint8_t>(color.y * 255.0f);
        const auto b = static_cast<uint8_t>(color.z * 255.0f);

        return (0xFF << 24) | (b << 16) | (g << 8) | r;
    }
}
