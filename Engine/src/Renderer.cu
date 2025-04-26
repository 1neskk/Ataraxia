#include <algorithm>
#include <iostream>
#include <cfloat>

#include <device_launch_parameters.h>

#include "Random.h"
#include "Renderer.h"
#include "BRDF.h"

#include <glm/ext/scalar_constants.hpp>

Renderer::Renderer() : d_spheres_(), d_materials_(),d_lights_(), d_accumulation_(), h_imageData_(),
    d_imageData_(), m_frameIndex(1)
{}

Renderer::~Renderer()
{
    d_imageData_.Release();
    freeDeviceMemory();

    delete[] h_imageData_;
}

void Renderer::allocateDeviceMemory(const Scene& scene)
{
    std::vector<Sphere> collectedSpheres;
    traverseSceneGraph(scene.rootNode, glm::mat4(1.0f), collectedSpheres);

    for (auto& sphere : collectedSpheres)
    {
	    if (static_cast<uint32_t>(sphere.id) >= scene.materials.size())
	    {
			std::cerr << "Sphere ID out of bounds: " << sphere.id << "\n";
			sphere.id = 0;
	    }
    }

    m_numSpheres = collectedSpheres.size();

    d_spheres_.Release();
	d_spheres_ = CudaBuffer<Sphere>(m_numSpheres);
    d_spheres_.CopyFromHost(collectedSpheres.data(), m_numSpheres);

	m_numMaterials = scene.materials.size();
    
    d_materials_.Release();
	d_materials_ = CudaBuffer<Material>(m_numMaterials);
	d_materials_.CopyFromHost(scene.materials.data(), m_numMaterials);


	m_numLights = scene.lights.size();

    d_lights_.Release();
	d_lights_ = CudaBuffer<Light>(m_numLights);
    d_lights_.CopyFromHost(scene.lights.data(), m_numLights);
}

void Renderer::freeDeviceMemory()
{
	d_spheres_.Release();
	d_materials_.Release();
	d_lights_.Release();
	d_accumulation_.Release();
}

void Renderer::traverseSceneGraph(const std::shared_ptr<SceneNode>& node, const glm::mat4& parentTransform, std::vector<Sphere>& spheres)
{
    glm::mat4 globalTransform = parentTransform;
    if (node)
    {
        node->updateGlobalTransform(parentTransform);
        globalTransform = node->getGlobalTransform();

        for (const auto& sphere : node->getSpheres())
        {
            Sphere transformedSphere = sphere;
            glm::vec4 transformedCenter = globalTransform * glm::vec4(sphere.center.x, sphere.center.y, sphere.center.z, 1.0f);
            transformedSphere.center = glm::vec3(transformedCenter) / transformedCenter.w;

            glm::vec3 scale;
            scale.x = glm::length(glm::vec3(globalTransform[0]));
            scale.y = glm::length(glm::vec3(globalTransform[1]));
            scale.z = glm::length(glm::vec3(globalTransform[2]));
            float uniformScale = (scale.x + scale.y + scale.z) / 3.0f;
            transformedSphere.radius *= uniformScale;

            spheres.emplace_back(transformedSphere);
        }

        for (const auto& child : node->getChildren())
        {
            traverseSceneGraph(child, globalTransform, spheres);
        }
    }
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

    try
    {
        h_imageData_ = new uint32_t[static_cast<uint64_t>(width) * height];
    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Failed to allocate host image data: " << e.what() << "\n";
        return;
    }

	d_imageData_.Release();
	d_imageData_ = CudaBuffer<uint32_t>(static_cast<uint64_t>(width) * height);

    d_accumulation_.Release();
	d_accumulation_ = CudaBuffer<glm::vec4>(static_cast<uint64_t>(width) * height);

    m_width = width;
    m_height = height;
    m_frameIndex = 1;
}

namespace
{
    __global__ void kernelRender(uint32_t width, uint32_t height, uint32_t* imageData, const Sphere* spheres,
        size_t numSpheres, const DeviceCamera d_camera, const Material* materials, size_t numMaterials, glm::vec4* accumulation,
        uint32_t frameIndex, const Light* lights, size_t numLights, Settings settings)
    {
        if (width == 0 || height == 0 || d_camera.width == 0 || d_camera.height == 0)
            return;

        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            const glm::vec4 color = Renderer::perPixel(x, y, width, spheres, numSpheres, d_camera, materials, numMaterials, frameIndex,
                lights, numLights, settings);
            const uint32_t pixelIndex = x + y * width;
            accumulation[pixelIndex] += color;
            glm::vec4 finalColor = accumulation[pixelIndex] / static_cast<float>(frameIndex);
            finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
            imageData[pixelIndex] = colorUtils::vec4ToRGBA(finalColor);
        }
    }
}

void Renderer::Render(Camera& camera, const Scene& scene)
{
    if (m_scene != &scene || m_frameIndex == 1)
    {
        m_scene = &scene;
        allocateDeviceMemory(scene);
    }

    if (m_frameIndex == 1)
        cudaMemset(d_accumulation_.GetData(), 0, static_cast<uint64_t>(m_width) * m_height * sizeof(glm::vec4));

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

    std::vector<Sphere> collectedSpheres;
    traverseSceneGraph(scene.rootNode, glm::mat4(1.0f), collectedSpheres);

    kernelRender<<<gridSize, blockSize>>>(m_width, m_height, d_imageData_.GetData(), d_spheres_.GetData(), m_numSpheres, d_camera,
        d_materials_.GetData(), m_numMaterials, d_accumulation_.GetData(), m_frameIndex, d_lights_.GetData(), m_numLights, m_settings);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << "\n";
        return;
    }

	err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
		std::cerr << "CUDA kernel synchronization error: " << cudaGetErrorString(err) << "\n";
		return;
    }

	d_imageData_.CopyToHost(h_imageData_, static_cast<size_t>(m_width) * m_height);

    m_image->setData(h_imageData_);
    Camera::freeDevice(d_camera);

    if (m_settings.accumulation)
        m_frameIndex++;
    else
        m_frameIndex = 1;
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
				// skyLight setting just adds a simple gradient to the background. It is not an actual light source.
                glm::vec3 missColor(0.6f, 0.7f, 0.9f);
                color += missColor * throughput;
            }
            break;
        }

        const auto& [center, radius, id] = spheres[ht.id];

        uint32_t materialIndex = id;
		if (materialIndex >= numMaterials)
			materialIndex = 0;

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
            uint32_t lightIndex = Random::Random::PcgHash(seed) % numLights;
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
