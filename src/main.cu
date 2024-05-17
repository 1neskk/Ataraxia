#include <cuda_runtime.h>

#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

#include "../thirdparty/glm/glm/glm.hpp"
#include "../thirdparty/glm/glm/gtc/matrix_transform.hpp"
#include "../thirdparty/glm/glm/gtc/type_ptr.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../thirdparty/stb/stb_image_write.h"

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Sphere
{
    glm::vec3 center;
    float radius;
};

__device__ bool intersect(const Ray& ray, const Sphere& sphere, float& t)
{
    const glm::vec3 oc = ray.origin - sphere.center; // vector from sphere center to ray origin
    const float a = glm::dot(ray.direction, ray.direction);
    const float b = 2.0f * glm::dot(oc, ray.direction);
    const float c = glm::dot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
    {
        return false;
    }
    else
    {
        const float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
        const float t2 = (-b + sqrt(discriminant)) / (2.0f * a);

        if (t1 > 0.0f)
        {
            t = t1;
            return true;
        }
        else if (t2 > 0.0f)
        {
            t = t2;
            return true;
        }
        return false;
    }
}

// CUDA kernel
__global__ void renderKernel(glm::vec3* image, int width, int height, Sphere* spheres, int numSpheres,
        glm::vec3 camPos, glm::vec3 camDir, glm::vec3 camUp, float fov)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    float scale = tan(fov * 0.5f * M_PI / 180.0f);
    float u = (2.0f * (x + 0.5f) / static_cast<float>(width) - 1.0f) * aspect_ratio * scale;
    float v = (1.0f - 2.0f * (y + 0.5f) / static_cast<float>(height)) * scale;

    glm::vec3 ray_origin = camPos;
    glm::vec3 ray_direction = glm::normalize(u * glm::normalize(glm::cross(camDir, camUp)) + v * camUp + camDir);

    Ray ray;
    ray.origin = ray_origin;
    ray.direction = ray_direction;

    float t;
    bool hit = false;

    for (int i = 0; i < numSpheres; i++)
    {
        if (intersect(ray, spheres[i], t))
        {
            hit = true;
            break;
        }
    }

    if (hit)
    {
        image[idx] = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        image[idx] = glm::vec3(0.0f, 0.0f, 0.0f);
    }
}

int main()
{
    constexpr int width = 800;
    constexpr int height = 600;
    constexpr size_t imageSize = width * height * sizeof(glm::vec3);

    auto* h_image = static_cast<glm::vec3*>(malloc(imageSize));
    glm::vec3* d_image;
    cudaMalloc((void**)&d_image, imageSize);

    Sphere h_spheres[1];
    h_spheres[0].center = glm::vec3(0.0f, 0.0f, -5.0f);
    h_spheres[0].radius = 1.0f;

    Sphere* d_spheres;
    cudaMalloc((void**)&d_spheres, sizeof(h_spheres));
    cudaMemcpy(d_spheres, h_spheres, sizeof(h_spheres), cudaMemcpyHostToDevice);
    
    // Camera params
    const glm::vec3 camPos = glm::vec3(0.0f, 0.0f, 0.0f);
    const glm::vec3 camDir = glm::normalize(glm::vec3(0.0f, 0.0f, -1.0f));
    const glm::vec3 camUp = glm::vec3(0.0f, 1.0f, 0.0f);
    constexpr float fov = 90.0f;

    // Debugging: Print out the sphere and camera data
    std::cout << "Sphere center: " << h_spheres[0].center.x << ", " << h_spheres[0].center.y << ", " << h_spheres[0].center.z << std::endl;
    std::cout << "Sphere radius: " << h_spheres[0].radius << std::endl;

    std::cout << "Camera position: " << camPos.x << ", " << camPos.y << ", " << camPos.z << std::endl;
    std::cout << "Camera direction: " << camDir.x << ", " << camDir.y << ", " << camDir.z << std::endl;
    std::cout << "Camera up: " << camUp.x << ", " << camUp.y << ", " << camUp.z << std::endl;
    std::cout << "Field of view: " << fov << std::endl;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    renderKernel<<<gridSize, blockSize>>>(d_image, width, height, d_spheres, 1, camPos, camDir, camUp, fov);

    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

    // Debugging: Check if the image data is correctly generated
    int hitCount = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int idx = y * width + x;
            const glm::vec3 color = h_image[idx];
            if (color.r > 0.0f)
                hitCount++;
        }
    }
    std::cout << "Number of hits: " << hitCount << std::endl;

    const auto image_data = static_cast<unsigned char*>(malloc(width * height * 4));
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int idx = y * width + x;
            const glm::vec3 color = h_image[idx];
            image_data[4 * idx + 0] = static_cast<unsigned char>(color.r * 255.0f);
            image_data[4 * idx + 1] = static_cast<unsigned char>(color.g * 255.0f);
            image_data[4 * idx + 2] = static_cast<unsigned char>(color.b * 255.0f);
            image_data[4 * idx + 3] = 255; // Alpha
        }
    }

    stbi_write_png("output.png", width, height, 4, image_data, width * 4);
    std::cout << "Image saved to 'output.png'" << std::endl;

    cudaFree(d_image);
    cudaFree(d_spheres);

    free(h_image);
    free(image_data);

    return 0;
}
