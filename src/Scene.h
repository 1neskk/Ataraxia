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

struct Material
{
	glm::vec3 albedo;
};

struct Light
{
	glm::vec3 position;
	glm::vec3 intensity;
};

struct Sphere
{
    glm::vec3 center;
    float radius;
    int id = 0;
	//Material material;
};

struct Scene
{
	//std::vector<Light> lightSources;
	std::vector<Sphere> spheres;
};
