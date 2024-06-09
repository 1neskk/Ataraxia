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

	int id = 0;
};

struct Light
{
	glm::vec3 position;
	glm::vec3 intensity;
};

struct Scene
{
	std::vector<Sphere> spheres;
	std::vector<Material> materials;
};
