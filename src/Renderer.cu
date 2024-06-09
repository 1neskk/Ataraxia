#include <algorithm>
#include <iostream>
#include <cfloat>

#include "Renderer.h"

Renderer::Renderer() : d_spheres_(nullptr), h_imageData_(nullptr), d_imageData_(nullptr)
{}

Renderer::~Renderer()
{
    if (d_imageData_)
        cudaFree(d_imageData_);

    if (d_spheres_)
        freeDeviceMemory();

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
    {
        m_image = std::make_shared<Image>(width, height, ImageType::RGBA);
    }

    delete[] h_imageData_;
    h_imageData_ = new uint32_t[width * height];

    if (d_imageData_)
        cudaFree(d_imageData_);

    cudaError_t err = cudaMalloc(&d_imageData_, width * height * sizeof(uint32_t));
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";

    m_width = width;
    m_height = height;
}

void Renderer::Render(Camera& camera, const Scene& scene)
{
    m_scene = &scene;

	allocateDeviceMemory(scene);

    if (!m_image)
        return;

	DeviceCamera d_camera;
    camera.allocateDevice(d_camera);

    dim3 blockSize(16, 16);
    dim3 numBlocks((m_width + blockSize.x - 1) / blockSize.x,
                   (m_height + blockSize.y - 1) / blockSize.y);

	kernelRender<<<numBlocks, blockSize>>>(m_width, m_height, d_imageData_, d_spheres_, scene.spheres.size(), d_camera);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
		std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << "\n";
        return;
    }

    cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
        std::cerr << "CUDA kernel synchronization error: " << cudaGetErrorString(err) << "\n";
		return;
	}

    err = cudaMemcpy(h_imageData_, d_imageData_, m_width * m_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
	    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
		return;
	}

    m_image->setData(h_imageData_);
	camera.freeDevice(d_camera);
}

void Renderer::allocateDeviceMemory(const Scene& scene)
{
	if (d_spheres_)
		cudaFree(d_spheres_);

	size_t numSpheres = scene.spheres.size();
	cudaError_t err = cudaMalloc(&d_spheres_, numSpheres * sizeof(Sphere));
	if (err != cudaSuccess)
		std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";

	err = cudaMemcpy(d_spheres_, scene.spheres.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
}

void Renderer::freeDeviceMemory()
{
	if (d_spheres_)
	{
        cudaFree(d_spheres_);
        d_spheres_ = nullptr;
	}
}

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres,
    size_t numSpheres, const DeviceCamera d_camera)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
	{
		const glm::vec4 color = Renderer::perPixel(x, y, width, spheres, numSpheres, d_camera);
		imageData[x + y * width] = colorUtils::vec4ToRGBA(color);
    }
}

__device__ Renderer::HitRecord Renderer::traceRay(const Ray& ray, const Sphere* spheres, size_t numSpheres)
{
    int closestSphere = -1;
    float tmin = FLT_MAX;

	for (size_t i = 0; i < numSpheres; i++)
    {
        const auto& [center, radius, id] = spheres[i];

		glm::vec3 oc = ray.origin - center;

		const float a = glm::dot(ray.direction, ray.direction);
		const float b = 2.0f * glm::dot(oc, ray.direction);
		const float c = glm::dot(oc, oc) - radius * radius;
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
        return rayMiss(ray);

	return rayHit(ray, tmin, closestSphere, spheres);
}

__device__ glm::vec4 Renderer::perPixel(uint32_t x, uint32_t y, uint32_t width, const Sphere* spheres,
    size_t numSpheres, const DeviceCamera& d_camera)
{
    Ray ray;
    ray.origin = d_camera.position;
    ray.direction = d_camera.rayDirection[x + y * width];

    glm::vec3 color(0.0f);
    float m = 1.0f;
    constexpr int bounces = 1;

    for (int i = 0; i < bounces; i++)
    {
		HitRecord ht = traceRay(ray, spheres, numSpheres);
        if (ht.t < 0.0f)
        {
			glm::vec3 missColor(0.6f, 0.7f, 0.9f);
			color += missColor * m;
            break;
        }

        glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
		float light = glm::dot(-lightDir, ht.normal);
		if (light < 0.0f)
			light = 0.0f;

        const auto& [center, radius, id] = spheres[ht.id];
		//const Material& mat = m_scene->materials[id];

        glm::vec3 sphereColor = glm::vec3(1.0f, 0.0f, 0.0f) * light;
		color += sphereColor * m;

        m *= 0.5f;

		ray.origin = ht.worldNormal + ht.normal * 0.0001f;
		ray.direction = glm::reflect(ray.direction, ht.normal * glm::vec3(0.5f, 0.5f, 0.5f));
    }
    return { color, 1.0f };
}

__device__ Renderer::HitRecord Renderer::rayMiss(const Ray& ray)
{
    HitRecord ht;
	ht.t = -1.0f;
    return ht;
}

__device__ Renderer::HitRecord Renderer::rayHit(const Ray& ray, float tmin, int index, const Sphere* spheres)
{
    HitRecord ht;
    ht.t = tmin;
    ht.id = index;

	const glm::vec3 origin = ray.origin - spheres[index].center;
    ht.worldNormal = origin + ray.direction * tmin;
	ht.normal = glm::normalize(ht.worldNormal);

	ht.worldNormal += spheres[index].center;

    return ht;
}