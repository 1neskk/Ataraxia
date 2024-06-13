#pragma once

#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>

namespace Random
{
	class Random
	{
	public:
		__device__ static void Init(curandState* state, unsigned long long seed, int id)
		{
			curand_init(seed, id, 0, state);
		}

		__device__ static uint32_t UInt(curandState* state, int id)
		{
			return curand(&state[id]);
		}

		__device__ static uint32_t UInt(curandState* state, uint32_t min, uint32_t max, int id)
		{
			return min + (curand(&state[id]) % (max - min + 1));
		}

		__device__ static float Float(curandState* state, int id)
		{
			return curand_uniform(&state[id]);
		}

		__device__ static float Float(curandState* state, float min, float max, int id)
		{
			return min + (max - min) * curand_uniform(&state[id]);
		}

		__device__ static glm::vec3 Vec3(curandState* state, int id)
		{
			return { Float(state, id), Float(state, id), Float(state, id) };
		}

		__device__ static glm::vec3 Vec3(curandState* state, float min, float max, int id)
		{
			return { Float(state, min, max, id), Float(state, min, max, id), Float(state, min, max, id) };
		}

		__device__ static glm::vec3 InUnitSphere(curandState* state, int id)
		{
			return glm::normalize(Vec3(state, -1.0f, 1.0f, id));
		}

		__device__ static glm::vec3 PcgInUnitSphere(uint32_t& seed)
		{
			return glm::normalize(glm::vec3(PcgFloat(seed) * 2.0f - 1.0f, PcgFloat(seed) * 2.0f - 1.0f, PcgFloat(seed) * 2.0f - 1.0f));
		}

		__device__ static uint32_t PcgHash(uint32_t& seed)
		{
			uint32_t state = seed * 747796405u + 2891336453u;
			uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
			return (word >> 22u) ^ word;
		}

		__device__ static float PcgFloat(uint32_t& seed)
		{
			seed = PcgHash(seed);
			return static_cast<float>(seed) / static_cast<float>(UINT32_MAX);
		}
	};
}