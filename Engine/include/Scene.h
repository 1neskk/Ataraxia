#pragma once

#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <vector>
#include "Camera.h"

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

	Sphere() = default;
	Sphere(const glm::vec3& c, float r, int materialId)
		: center(c), radius(r), id(materialId) {
	}
};

struct Light
{
	glm::vec3 position;
	glm::vec3 color;
	float intensity;

	Light() = default;
	Light(const glm::vec3& pos, const glm::vec3& col, float i)
		: position(pos), color(col), intensity(i) {
	}
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

	Material() = default;
};

struct Settings
{
	bool accumulation = true;
	bool skyLight = false;
	int maxBounces = 15;

	Settings() = default;
};

struct Scene
{
	std::vector<Sphere> spheres;
	std::vector<Material> materials;
	std::vector<Light> lights;
	Settings settings;
	Camera camera;

	Scene() = default;
	Scene(const Scene& other) = default;
	Scene& operator=(const Scene& other) = default;
	Scene(Scene&& other) = default;
	Scene& operator=(Scene&& other) = default;
};
