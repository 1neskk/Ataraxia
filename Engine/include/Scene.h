#pragma once

#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <vector>

struct Ray
{
	glm::vec3 origin;
    glm::vec3 direction;
};

struct Sphere
{
	glm::vec3 center;
	float radius;
	int id = 0;
};

struct Light
{
	glm::vec3 position;
	glm::vec3 color;
	float intensity;
};

struct Material
{
	glm::vec3 albedo{ 1.0f };

	float roughness = 0.0f;
	float metallic = 0.0f;
	glm::vec3 F0{ 0.04f };

	glm::vec3 emissionColor{ 0.0f };
	float emissionIntensity = 0.0f;

	int id = 0;

	__host__ __device__ glm::vec3 getEmission() const { return emissionColor * emissionIntensity; }
};

struct Settings
{
	bool accumulation = true;
	bool skyLight = false;
	int maxBounces = 15;
};

struct Scene
{
	std::vector<Sphere> spheres;
	std::vector<Material> materials;
	std::vector<Light> lights;
};
