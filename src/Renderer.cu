#include "Renderer.h"

#include <algorithm>

Renderer::Renderer() : m_image(nullptr), h_imageData_(nullptr), d_imageData_(nullptr)
{}

Renderer::~Renderer()
{
    if (d_imageData_)
        cudaFree(d_imageData_);

    delete[] h_imageData_;
}


void Renderer::onResize(uint32_t width, uint32_t height)
{
    if (m_image)
    {
        if (m_image->getWidth() == width && m_image->getHeight() == height)
        {
            return;
        }
        m_image->resize(width, height);
    }
    else
        m_image = std::make_shared<Image>(width, height, ImageType::RGBA);

    delete[] h_imageData_;
    h_imageData_ = new uint32_t[width * height];

    if (d_imageData_)
        cudaFree(d_imageData_);

    cudaMalloc(&d_imageData_, width * height * sizeof(uint32_t));

    m_width = width;
    m_height = height;
}

void Renderer::Render(const Scene& scene)
{
    m_scene = &scene;

    if (!m_image)
        return;

    dim3 blockSize(16, 16);
    dim3 numBlocks((m_width + blockSize.x - 1) / blockSize.x,
                   (m_height + blockSize.y - 1) / blockSize.y);

    kernelRender<<<numBlocks, blockSize>>>(m_width, m_height, d_imageData_, m_scene->spheres, m_scene->numSpheres);

    cudaMemcpy(h_imageData_, d_imageData_, m_width * m_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    m_image->setData(h_imageData_);
}

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres, size_t numSpheres)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        Vec4 color = Renderer::perPixel(x, y, width, height, spheres, numSpheres);
        imageData[x + y * width] = colorUtils::vec4ToRGBA(color);
    }
}

__device__ bool Renderer::intersect(const Ray& ray, const Sphere* spheres, size_t numSpheres, float& t)
{
    int closestSphere = -1;
    float tmin = INFINITY;

    for (size_t i = 0; i < numSpheres; i++)
    {
        const auto& sphere = spheres[i];

        Vec3 origin = ray.origin - sphere.center;

        const float a = ray.direction.dot(ray.direction);
        const float b = 2.0f * origin.dot(ray.direction);
        const float c = origin.dot(origin) - sphere.radius * sphere.radius;
        const float discriminant = b * b - 4 * a * c;

        if (discriminant < 0.0f)
            continue;

        float t0 = (-b - sqrt(discriminant)) / (2.0f * a);
        float t1 = (-b + sqrt(discriminant)) / (2.0f * a);
        const float t = t0 < t1 ? t0 : t1;

        if (t > 0.0f && t < tmin)
        {
            tmin = t;
            closestSphere = static_cast<int>(i);
        }
    }

    if (closestSphere < 0)
        return false;
    
    t = tmin;
    return true;
}

__device__ Vec4 Renderer::perPixel(uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                   const Sphere* spheres, size_t numSpheres)
{
    const float aspectRatio = width / static_cast<float>(height);
    const float fov = 90.0f;
    const float scale = tan(fov * 0.5f * M_PI / 180.0f);

    const float u = (2.0f * (x + 0.5f) / width - 1.0f) * aspectRatio * scale;
    const float v = (1.0f - 2.0f * (y + 0.5f) / height) * scale;

    Vec3 direction = Vec3(u, v, -1.0f).normalize();
    Ray ray = { Vec3(0.0f, 0.0f, 0.0f), direction };

    float t;
    if (intersect(ray, spheres, numSpheres, t))
    {
        return Vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }

    return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
}
