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

struct Material
{
	glm::vec3 albedo{ 1.0f };
	float diffuse{ 1.0f };
	float specular{ 0.0f };
	float shininess{ 0.0f };

	glm::vec3 emissionColor{ 0.0f };
	float emissionIntensity = 0.0f;

	float reflection{ 0.0f };
	float transparency{ 0.0f };
	float ior{ 1.0f }; // index of refraction

	int id = 0;

	__host__ __device__ glm::vec3 getEmission() const { return emissionColor * emissionIntensity; }
};

struct Settings
{
	bool accumulation = true;
	bool skyLight = false;
};

struct Scene
{
	std::vector<Sphere> spheres;
	std::vector<Material> materials;
};
