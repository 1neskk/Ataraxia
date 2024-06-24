#include <algorithm>
#include <iostream>
#include <cfloat>

#include <device_launch_parameters.h>

#include "Random.h"
#include "Renderer.h"
#include "BRDF.h"

#include <glm/ext/scalar_constants.hpp>

Renderer::Renderer() : d_spheres_(nullptr), d_materials_(nullptr),d_lights_(nullptr), d_accumulation_(nullptr), h_imageData_(nullptr),
                       d_imageData_(nullptr), m_frameIndex(1)
{}

Renderer::~Renderer()
{
	cudaFree(d_imageData_);
	cudaFree(d_accumulation_);
	freeDeviceMemory();

    delete[] h_imageData_;
}

void Renderer::onResize(uint32_t width, uint32_t height)
{
	if (m_image && m_image->getWidth() == width && m_image->getHeight() == height)
		return;

	m_image = std::make_shared<Image>(width, height, ImageType::RGBA);

    delete[] h_imageData_;
    h_imageData_ = new uint32_t[width * height];

    cudaFree(d_imageData_);
    cudaMalloc(&d_imageData_, static_cast<unsigned long long>(width) * height * sizeof(uint32_t));

	cudaFree(d_accumulation_);
	cudaMalloc(&d_accumulation_, static_cast<unsigned long long>(width) * height * sizeof(glm::vec4));

    m_width = width;
    m_height = height;

    m_frameIndex = 1;
}

void Renderer::Render(Camera& camera, const Scene& scene)
{
    m_scene = &scene;

	allocateDeviceMemory(scene);

    if (m_frameIndex == 1)
		cudaMemset(d_accumulation_, 0, static_cast<unsigned long long>(m_width) * m_height * sizeof(glm::vec4));

    if (!m_image)
        return;

	DeviceCamera d_camera;
    camera.allocateDevice(d_camera);

    dim3 blockSize(16, 16);
    dim3 numBlocks((m_width + blockSize.x - 1) / blockSize.x,
                   (m_height + blockSize.y - 1) / blockSize.y);

	kernelRender<<<numBlocks, blockSize>>>(m_width, m_height, d_imageData_, d_spheres_, scene.spheres.size(), d_camera,
        d_materials_, scene.materials.size(), d_accumulation_, m_frameIndex, d_lights_, scene.lights.size());

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

    if (m_settings.accumulation)
        m_frameIndex++;
    else
        m_frameIndex = 1;
}

void Renderer::allocateDeviceMemory(const Scene& scene)
{
	cudaFree(d_spheres_);
	cudaMalloc(&d_spheres_, scene.spheres.size() * sizeof(Sphere));
	cudaMemcpy(d_spheres_, scene.spheres.data(), scene.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
	

	cudaFree(d_materials_);
	cudaMalloc(&d_materials_, scene.materials.size() *sizeof(Material));
    cudaMemcpy(d_materials_, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaFree(d_lights_);
	cudaMalloc(&d_lights_, scene.lights.size() * sizeof(Light));
	cudaMemcpy(d_lights_, scene.lights.data(), scene.lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
}

void Renderer::freeDeviceMemory()
{
	cudaFree(d_spheres_);
	d_spheres_ = nullptr;

    cudaFree(d_materials_);
	d_materials_ = nullptr;

	cudaFree(d_lights_);
	d_lights_ = nullptr;

	cudaFree(d_accumulation_);
	d_accumulation_ = nullptr;
}

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres,
    size_t numSpheres, const DeviceCamera d_camera, const Material* materials, size_t numMaterials, glm::vec4* accumulation,
	uint32_t frameIndex, const Light* lights, size_t numLights)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
	{
		const glm::vec4 color = Renderer::perPixel(x, y, width, spheres, numSpheres, d_camera, materials, numMaterials, frameIndex, lights,
			numLights);
		const uint32_t pixelIndex = x + y * width;
        accumulation[pixelIndex] += color;
		glm::vec4 finalColor = accumulation[pixelIndex] / static_cast<float>(frameIndex);
        finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
		imageData[pixelIndex] = colorUtils::vec4ToRGBA(finalColor);
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
    size_t numSpheres, const DeviceCamera& d_camera, const Material* materials, size_t numMaterials, uint32_t frameIndex,
	const Light* lights, size_t numLights)
{
	__shared__ Light sharedLights[10];

	if (threadIdx.x < numLights)
		sharedLights[threadIdx.x] = lights[threadIdx.x];

	__syncthreads();

	Ray ray;
	ray.origin = d_camera.position;
	ray.direction = d_camera.rayDirection[x + y * width];

    glm::vec3 color(0.0f);
    glm::vec3 throughput(1.0f);

	uint32_t seed = x + y * width;
	seed *= frameIndex;

	constexpr int bounces = 5; // Max bounces

	for (int i = 0; i < bounces; i++)
	{
		seed += i;

		HitRecord ht = traceRay(ray, spheres, numSpheres);
		if (ht.t < 0.0f)
		{
#define SKY_LIGHT 0
#if SKY_LIGHT
			glm::vec3 missColor(0.6f, 0.7f, 0.9f);
			color += missColor * throughput;
			break;
#else
			color += glm::vec3(0.0f);
			break;
#endif
		}

		const auto& [center, radius, id] = spheres[ht.id];
		const Material* mat = &materials[id];

		glm::vec3 baseReflectivity = glm::mix(mat->F0, mat->albedo, mat->metallic);
		for (size_t j = 0; j < numLights && j < 10; j++)
		{
			const Light& light = sharedLights[j];
			glm::vec3 L = light.position - ht.worldPos;
			const float distance = glm::length(L);
			L = glm::normalize(L);

			Ray shadowRay;
			shadowRay.origin = ht.worldPos + ht.worldNormal * 0.0001f;
			shadowRay.direction = L;

			HitRecord shadowHt = traceRay(shadowRay, spheres, numSpheres);
			if (shadowHt.t > 0.0f && shadowHt.t < distance)
				continue;

			const glm::vec3 V = -ray.direction;
			const glm::vec3 N = ht.worldNormal;

			const glm::vec3 diffuse = BRDF::lambertian(mat->albedo);
			const glm::vec3 specular = BRDF::cookTorrance(mat->albedo, baseReflectivity, mat->metallic, mat->roughness, N, V, L);

			const glm::vec3 brdf = diffuse + specular;
			const float NdotL = glm::max(glm::dot(N, L), 0.0f);

			color += light.intensity * brdf * NdotL * throughput;
		}

		throughput *= mat->albedo;

		ray.origin = ht.worldPos + ht.worldNormal * 0.0001f;

		if (mat->metallic > 0.0f)
			ray.direction = BRDF::sampleGGX(ht.worldNormal, mat->roughness, seed);
		else
			ray.direction = BRDF::sampleHemisphereCosineWeighted(ht.worldNormal, seed);
	}
    return { color, 1.0f };
}

__device__ Renderer::HitRecord Renderer::rayMiss(const Ray& ray)
{
    HitRecord ht;
	ht.t = -1.0f;
    return ht;
}

__device__ Renderer::HitRecord Renderer::rayHit(const Ray& ray, float tmin, const int index, const Sphere* spheres)
{
    HitRecord ht;
    ht.t = tmin;
    ht.id = index;

	const glm::vec3 origin = ray.origin - spheres[index].center;
    ht.worldPos = origin + ray.direction * tmin;
	ht.worldNormal = glm::normalize(ht.worldPos);

	ht.worldPos += spheres[index].center;

    return ht;
}