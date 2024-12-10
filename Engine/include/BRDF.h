#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "Random.h"

class BRDF
{
public:
    __device__ static glm::vec3 lambertian(const glm::vec3& albedo);
    __device__ static glm::vec3 cookTorrance(const glm::vec3& albedo, const glm::vec3& F0, float metallic, float roughness, const glm::vec3& N, const glm::vec3& V, const glm::vec3& L);

    __device__ static glm::vec3 fresnelSchlick(const glm::vec3& F0, float cosTheta);

    __device__ static float distributionGGX(float NdotH, float roughness);
    __device__ static float geometrySchlickGGX(float NdotV, float roughness);
    __device__ static float geometrySmith(float NdotV, float NdotL, float roughness);

	// Sampling
    __device__ static glm::vec3 sampleHemisphereCosineWeighted(const glm::vec3& N, uint32_t& seed);
	__device__ static glm::vec3 sampleGGX(const glm::vec3& N, float roughness, uint32_t& seed);
};