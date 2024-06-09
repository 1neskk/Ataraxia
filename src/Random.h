#pragma once

#include <random>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

namespace Random
{
	class Random
	{
	public:
		static void Init()
		{
			s_RandomEngine.seed(std::random_device()());
		}

		static uint32_t UInt()
		{
			return s_Distribution(s_RandomEngine);
		}

		static uint32_t UInt(uint32_t min, uint32_t max)
		{
			return min + (s_Distribution(s_RandomEngine) % (max - min + 1));
		}

		static float Float()
		{
			return static_cast<float>(s_Distribution(s_RandomEngine)) / static_cast<float>(std::numeric_limits<uint32_t>::max());
		}

		static glm::vec3 Vec3()
		{
			return { Float(), Float(), Float() };
		}

		static glm::vec3 Vec3(float min, float max)
		{
			return { Float() * (max - min) + min, Float() * (max - min) + min, Float() * (max - min) + min };
		}

		static glm::vec3 InUnitSphere()
		{
			return glm::normalize(Vec3(-1.0f, 1.0f));
		}

		static glm::vec3 PcgInUnitSphere(uint32_t& seed)
		{
			return glm::normalize(glm::vec3(PcgFloat(seed) * 2.0f - 1.0f, PcgFloat(seed) * 2.0f - 1.0f, PcgFloat(seed) * 2.0f - 1.0f));
		}

		static uint32_t PcgHash(uint32_t& seed)
		{
			uint32_t state = seed * 747796405u + 2891336453u;
			uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
			return (word >> 22u) ^ word;
		}

		static float PcgFloat(uint32_t& seed)
		{
			seed = PcgHash(seed);
			return static_cast<float>(seed) / static_cast<float>(std::numeric_limits<uint32_t>::max());
		}
	private:
		static thread_local std::mt19937 s_RandomEngine;
		static std::uniform_int_distribution<std::mt19937::result_type> s_Distribution;
	};

	thread_local std::mt19937 Random::s_RandomEngine;
	std::uniform_int_distribution<std::mt19937::result_type> Random::s_Distribution;
}