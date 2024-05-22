#include "main.h"

__device__ bool intersect(const Ray& ray, const Sphere& sphere, float& t)
{
    const Vec3 oc = ray.origin - sphere.center;
    const float a = Vec3::dot(ray.direction, ray.direction);
    const float b = 2.0f * Vec3::dot(oc, ray.direction);
    const float c = Vec3::dot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
    {
        t = (-b + sqrtf(discriminant)) / (2.0f * a);
        return t >= 0.0f;
    }
    else
    {
        t = (-b - sqrtf(discriminant)) / (2.0f * a);
        return t > 0.0f;
    }
}

// CUDA kernel
__global__ void renderKernel(Vec3* image, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
	float u = (static_cast<float>(x) / static_cast<float>(width)) * 2.0f - 1.0f;
    const float v = (static_cast<float>(y) / static_cast<float>(height)) * 2.0f - 1.0f;

    const Ray ray = { Vec3(0.0f, 0.0f, 0.0f), Vec3(u * aspectRatio, v, -1.0f).normalize() };
    const Sphere sphere = { Vec3(0.0f, 0.0f, -3.0f), 1.0f };

    float t = INFINITY;
    if (intersect(ray, sphere, t))
    {
        image[idx] = Vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        image[idx] = Vec3(0.0f, 0.0f, 0.0f);
    }
}

int main()
{
    constexpr int width = 1200;
    constexpr int height = 800;
    constexpr size_t imageSize = width * height * sizeof(Vec3);

    auto* h_image = static_cast<Vec3*>(malloc(imageSize));
    Vec3* d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), imageSize);

    Sphere h_spheres[1];
    h_spheres[0].center = Vec3(0.0f, 0.0f, 0.0f);
    h_spheres[0].radius = 1.0f;

    Sphere* d_spheres;
    cudaMalloc(reinterpret_cast<void**>(&d_spheres), sizeof(h_spheres));
    cudaMemcpy(d_spheres, h_spheres, sizeof(h_spheres), cudaMemcpyHostToDevice);

    // Debugging: Print out the sphere data
    std::cout << "Sphere center: " << h_spheres[0].center.x << ", " << h_spheres[0].center.y << ", " << h_spheres[0].center.z << std::endl;
    std::cout << "Sphere radius: " << h_spheres[0].radius << std::endl;

    dim3 blocks(16, 16);
    dim3 grids((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1) / blocks.y);
    renderKernel<<<grids, blocks >>>(d_image, width, height);

    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

    // Debugging: Check if the image data is correctly generated
    int hitCount = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int idx = y * width + x;
            const Vec3 color = h_image[idx];
            if (color.x > 0.0f)
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
            const Vec3 color = h_image[idx];
            image_data[4 * idx + 0] = static_cast<unsigned char>(color.x * 255.0f);
            image_data[4 * idx + 1] = static_cast<unsigned char>(color.y * 255.0f);
            image_data[4 * idx + 2] = static_cast<unsigned char>(color.z * 255.0f);
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
