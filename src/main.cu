#include "main.h"

__device__ bool intersect(const Ray& ray, const Sphere& sphere, float& t)
{
    const Vec3 oc = ray.origin - sphere.center;
    const float a = ray.direction.dot(ray.direction);
    const float b = 2.0f * oc.dot(ray.direction);
    const float c = oc.dot(oc) - sphere.radius * sphere.radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
    {
        return false;
    }
    else
    {
        const float sqrtDiscriminant = sqrtf(discriminant);
        const float t1 = (-b - sqrtDiscriminant) / (2.0f * a);
        const float t2 = (-b + sqrtDiscriminant) / (2.0f * a);
        if (t1 > 0.0f)
        {
            t = t1;
            return true;
        }
        if (t2 > 0.0f)
        {
            t = t2;
            return true;
        }
        return false;
    }
}

// CUDA kernel
__global__ void renderKernel(Vec3* image, int width, int height, const Sphere* spheres, int numSpheres)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    float fov = tanf(M_PI / 4.0f); // 45 degrees
    float u = ((2.0f * (x + 0.5f) / static_cast<float>(width)) - 1.0f) * aspectRatio * fov;
    float v = (1.0f - 2.0f * (y + 0.5f) / static_cast<float>(height)) * fov;

    const Ray ray = { Vec3(0.0f, 0.0f, 0.0f), Vec3(u, v, -1.0f).normalize() };

    float t = INFINITY;
    Vec3 color = Vec3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < numSpheres; i++)
    {
        if (intersect(ray, spheres[i], t))
        {
            Vec3 hitPoint = ray.origin + ray.direction * t;
            Vec3 normal = (hitPoint - spheres[i].center).normalize();
            color = (normal + Vec3(1.0f, 1.0f, 1.0f)) * 0.5f;
            break;
        }
    }
    image[idx] = color;
}

int main()
{
    constexpr int width = 1200;
    constexpr int height = 800;
    constexpr size_t imageSize = width * height * sizeof(Vec3);
    constexpr int numSpheres = 3;

    auto* h_image = static_cast<Vec3*>(malloc(imageSize));
    Vec3* d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), imageSize);

    Sphere h_spheres[3];
    h_spheres[0] = { Vec3(0.0f, 0.0f, -3.0f), 0.5f };
    h_spheres[1] = { Vec3(-3.0f, 0.0f, -3.0f), 0.5f };
    h_spheres[2] = { Vec3(3.0f, 0.0f, -3.0f), 0.5f };
    Sphere* d_spheres;
    cudaMalloc(reinterpret_cast<void**>(&d_spheres), sizeof(h_spheres));
    cudaMemcpy(d_spheres, h_spheres, sizeof(h_spheres), cudaMemcpyHostToDevice);

    dim3 blocks(16, 16);
    dim3 grids((width + blocks.x - 1) / blocks.x, (height + blocks.y - 1) / blocks.y);
    renderKernel<<<grids, blocks>>>(d_image, width, height, d_spheres, numSpheres);

    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

    int hitCount = 0;
    const auto image_data = static_cast<unsigned char*>(malloc(width * height * 4));
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int idx = y * width + x;
            Vec3 color = h_image[idx];
            if (color.x > 0.0f)
                hitCount++;

            image_data[4 * idx + 0] = static_cast<unsigned char>(color.x * 255.0f);
            image_data[4 * idx + 1] = static_cast<unsigned char>(color.y * 255.0f);
            image_data[4 * idx + 2] = static_cast<unsigned char>(color.z * 255.0f);
            image_data[4 * idx + 3] = 255; // Alpha
        }
    }

    stbi_write_png("output.png", width, height, 4, image_data, width * 4);
    std::cout << "Image saved to 'output.png'" << std::endl;
    std::cout << "Hit count: " << hitCount << std::endl;

    cudaFree(d_image);
    cudaFree(d_spheres);

    free(h_image);
    free(image_data);

    return 0;
}
