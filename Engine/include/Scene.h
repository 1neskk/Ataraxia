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

	int id = 0;
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
	std::vector<Light> lights;
};
