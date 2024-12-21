#include <algorithm>
#include <iostream>
#include <cfloat>

#include <device_launch_parameters.h>

#include "Random.h"
#include "Renderer.h"
#include "BRDF.h"

#include <glm/ext/scalar_constants.hpp>

#ifdef _DEBUG
#define CUDA_CHECK(call) \
	do \
	{ \
		cudaError_t err = call; \
		if (err != cudaSuccess) \
		{ \
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
			std::exit(EXIT_FAILURE); \
		} \
	} while (0)
#endif

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres,
	size_t numSpheres, const DeviceCamera d_camera, const Material* materials, size_t numMaterials, glm::vec4* accumulation,
	uint32_t frameIndex, const Light* lights, size_t numLights, Settings settings);

Renderer::Renderer() : d_spheres_(nullptr), d_materials_(nullptr),d_lights_(nullptr), d_accumulation_(nullptr), h_imageData_(nullptr),
                       d_imageData_(nullptr), m_frameIndex(1)
{}

Renderer::~Renderer()
{
	cudaFree(d_imageData_);
	freeDeviceMemory();

    delete[] h_imageData_;
}

void Renderer::allocateDeviceMemory(const Scene& scene)
{
#ifdef _DEBUG
	CUDA_CHECK(cudaFree(d_spheres_));
	CUDA_CHECK(cudaMalloc(&d_spheres_, scene.spheres.size() * sizeof(Sphere)));
	CUDA_CHECK(cudaMemcpy(d_spheres_, scene.spheres.data(), scene.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
	
	CUDA_CHECK(cudaFree(d_materials_));
	CUDA_CHECK(cudaMalloc(&d_materials_, scene.materials.size() *sizeof(Material)));
	CUDA_CHECK(cudaMemcpy(d_materials_, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaFree(d_lights_));
	CUDA_CHECK(cudaMalloc(&d_lights_, scene.lights.size() * sizeof(Light)));
	CUDA_CHECK(cudaMemcpy(d_lights_, scene.lights.data(), scene.lights.size() * sizeof(Light), cudaMemcpyHostToDevice));
#else
	cudaFree(d_spheres_);
	cudaMalloc(&d_spheres_, scene.spheres.size() * sizeof(Sphere));
	cudaMemcpy(d_spheres_, scene.spheres.data(), scene.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

	cudaFree(d_materials_);
	cudaMalloc(&d_materials_, scene.materials.size() * sizeof(Material));
	cudaMemcpy(d_materials_, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaFree(d_lights_);
	cudaMalloc(&d_lights_, scene.lights.size() * sizeof(Light));
	cudaMemcpy(d_lights_, scene.lights.data(), scene.lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
#endif

}

void Renderer::freeDeviceMemory()
{
#ifdef _DEBUG
	CUDA_CHECK(cudaFree(d_spheres_));
	d_spheres_ = nullptr;

	CUDA_CHECK(cudaFree(d_materials_));
	d_materials_ = nullptr;

	CUDA_CHECK(cudaFree(d_lights_));
	d_lights_ = nullptr;

	CUDA_CHECK(cudaFree(d_accumulation_));
	d_accumulation_ = nullptr;
#else
	cudaFree(d_spheres_);
	d_spheres_ = nullptr;

	cudaFree(d_materials_);
	d_materials_ = nullptr;

	cudaFree(d_lights_);
	d_lights_ = nullptr;

	cudaFree(d_accumulation_);
	d_accumulation_ = nullptr;
#endif
}

void Renderer::onResize(uint32_t width, uint32_t height)
{
	if (m_image && m_image->getWidth() == width && m_image->getHeight() == height)
		return;

#ifdef _DEBUG
	try
	{
		m_image = std::make_shared<Image>(width, height, ImageType::RGBA);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create image: " << e.what() << "\n";
		return;
	}

	if (!m_image)
	{
		std::cerr << "Failed to create image" << "\n";
		return;
	}

#else
	m_image = std::make_shared<Image>(width, height, ImageType::RGBA);
#endif

	delete[] h_imageData_;
	h_imageData_ = nullptr;

#ifdef _DEBUG
	try
	{
		h_imageData_ = new uint32_t[width * height];
	}
	catch (const std::bad_alloc& e)
	{
		std::cerr << "Failed to allocate host image data: " << e.what() << "\n";
		return;
	}

	CUDA_CHECK(cudaFree(d_imageData_));
	cudaError_t err = cudaMalloc(&d_imageData_, static_cast<unsigned long long>(width) * height * sizeof(uint32_t));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device image data: " << cudaGetErrorString(err) << "\n";
		delete[] h_imageData_;
		h_imageData_ = nullptr;
		return;
	}

	CUDA_CHECK(cudaFree(d_accumulation_));
	err = cudaMalloc(&d_accumulation_, static_cast<unsigned long long>(width) * height * sizeof(glm::vec4));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate accumulation buffer: " << cudaGetErrorString(err) << "\n";
		cudaFree(d_imageData_);
		delete[] h_imageData_;
		h_imageData_ = nullptr;
		return;
	}
#else
	h_imageData_ = new uint32_t[width * height];

	cudaFree(d_imageData_);
	cudaMalloc(&d_imageData_, static_cast<unsigned long long>(width) * height * sizeof(uint32_t));

	cudaFree(d_accumulation_);
	cudaMalloc(&d_accumulation_, static_cast<unsigned long long>(width) * height * sizeof(glm::vec4));
#endif

	m_width = width;
	m_height = height;
	m_frameIndex = 1;
}

void Renderer::Render(Camera& camera, const Scene& scene)
{
    m_scene = &scene;

	allocateDeviceMemory(scene); // TODO: Change device memory allocation to not be done every frame.

    if (m_frameIndex == 1)
		cudaMemset(d_accumulation_, 0, static_cast<unsigned long long>(m_width) * m_height * sizeof(glm::vec4));

    if (!m_image)
        return;

	DeviceCamera d_camera;
    camera.allocateDevice(d_camera);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int minGridSize, optimalBlockSize;

#ifdef _DEBUG
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, kernelRender, 0, 0));

	std::cout << "\n======================================\n\n";
	std::cout << "Optimal block size: " << optimalBlockSize << "\n";
#else
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, kernelRender, 0, 0);
#endif

	int blockSizeX = optimalBlockSize / 32;
	int blockSizeY = optimalBlockSize / 32;

	dim3 blockSize(blockSizeX, blockSizeY);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

#ifdef _DEBUG
	if (blockSize.x * blockSize.y > optimalBlockSize)
	{
		std::cerr << "Block size exceeds optimal value; adjust accordingly.\n";
		return;
	}

	std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ")\n";
	std::cout << "Block size: (" << blockSize.x << ", " << blockSize.y << ") | " << blockSize.x * blockSize.y << " Threads per block\n";
#endif

	kernelRender<<<gridSize, blockSize>>>(m_width, m_height, d_imageData_, d_spheres_, scene.spheres.size(), d_camera,
        d_materials_, scene.materials.size(), d_accumulation_, m_frameIndex, d_lights_, scene.lights.size(), m_settings);

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
	Camera::freeDevice(d_camera);

    if (m_settings.accumulation)
        m_frameIndex++;
    else
        m_frameIndex = 1;
}

__global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres,
    size_t numSpheres, const DeviceCamera d_camera, const Material* materials, size_t numMaterials, glm::vec4* accumulation,
	uint32_t frameIndex, const Light* lights, size_t numLights, Settings settings)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
	{
		const glm::vec4 color = Renderer::perPixel(x, y, width, spheres, numSpheres, d_camera, materials, numMaterials, frameIndex, lights,
			numLights, settings);
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
	const Light* lights, size_t numLights, Settings settings)
{
	__shared__ Light sharedLights[10];

	const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
	if (tid < numLights && tid < 10)
		sharedLights[tid] = lights[tid];

	__syncthreads();

	Ray ray;
	ray.origin = d_camera.position;
	ray.direction = d_camera.rayDirection[x + y * width];

	glm::vec3 color(0.0f);
	// The throughput vector accounts for the attenuation of light as it bounces around the scene.
    glm::vec3 throughput(1.0f);

	uint32_t seed = x + y * width;
	seed *= frameIndex;

	int bounces = settings.maxBounces;
	for (int i = 0; i < bounces; i++)
	{
		seed += i;

		HitRecord ht = traceRay(ray, spheres, numSpheres);
		if (ht.t < 0.0f)
		{
			if (settings.skyLight)
			{
				glm::vec3 missColor(0.6f, 0.7f, 0.9f);
				color += missColor * throughput;
			}
			break;
		}

		const auto& [center, radius, id] = spheres[ht.id];
		const Material* mat = &materials[id];

		// Emission Logic
		if (mat->emissionIntensity > 0.0f)
		{
			const glm::vec3 emission = mat->getEmission();
			color += emission * throughput;
		}

		glm::vec3 baseReflectivity = glm::mix(mat->F0, mat->albedo, mat->metallic);

		// Light Sampling Logic (only supports point lights for now)
		if (numLights > 0)
		{
			int lightIndex = Random::Random::PcgHash(seed) % numLights;
			const Light& sampledLight = lights[lightIndex];

			glm::vec3 L = sampledLight.position - ht.worldPos;
			float distanceSquared = glm::dot(L, L);
			L = glm::normalize(L);

			Ray shadowRay;
			shadowRay.origin = ht.worldPos + ht.worldNormal * 0.0001f;
			shadowRay.direction = L;

			HitRecord shadowHt = traceRay(shadowRay, spheres, numSpheres);
			if (shadowHt.t > 0.0f && shadowHt.t * shadowHt.t < distanceSquared)
			{
				// Light is occluded; skip contribution.
			}
			else
			{
				// Evaluate BRDF where V is the view direction, N is the surface normal, and L is the light direction.
				glm::vec3 V = -ray.direction;
				glm::vec3 N = ht.worldNormal;
				glm::vec3 specular = BRDF::cookTorrance(mat->albedo, baseReflectivity, mat->metallic, mat->roughness, N, V, L);
				glm::vec3 emission = sampledLight.color * sampledLight.intensity;

				// Calculate PDF (Probability Density Function), for point lights, it's 1.
				float pdf = 1.0f;

				color += emission * specular * throughput / pdf;
			}
		}

		throughput *= mat->albedo;
		ray.origin = ht.worldPos + ht.worldNormal * 0.0001f;

		// Randomly terminate the ray with a probability based on the throughput length in order to prevent infinite loops.
		float p = glm::max(0.1f, glm::min(1.0f, glm::length(throughput)));
		if (Random::Random::PcgFloat(seed) > p)
			break;

		throughput /= p;

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