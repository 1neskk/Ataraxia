#include "BRDF.h"
#include "Random.h"

__device__ glm::vec3 BRDF::lambertian(const glm::vec3& albedo)
{
	return albedo / glm::pi<float>();
}

__device__ glm::vec3 BRDF::cookTorrance(const glm::vec3& albedo, const glm::vec3& F0, float metallic, float roughness,
	const glm::vec3& N, const glm::vec3& V, const glm::vec3& L)

{
	glm::vec3 H = glm::normalize(V + L);
	float NdotL = glm::max(glm::dot(N, L), 0.0000001f); // prevent division by zero
	float NdotV = glm::max(glm::dot(N, V), 0.0000001f);
	float NdotH = glm::max(glm::dot(N, H), 0.0f);
	float VdotH = glm::max(glm::dot(V, H), 0.0f);

	glm::vec3 F = fresnelSchlick(F0, VdotH);
	float D = distributionGGX(NdotH, roughness);
	float G = geometrySmith(NdotV, NdotL, roughness);

	glm::vec3 kS = F;
	glm::vec3 kD = glm::vec3(1.0f) - kS;
	kD *= 1.0f - metallic;

	glm::vec3 num = D * G * F;
	float denom = 4.0f * NdotL * NdotV + 0.001f;
	glm::vec3 specular = num / denom;

	glm::vec3 diffuse = (1.0f - F) * albedo / glm::pi<float>();

	return (kD * diffuse + specular) * NdotL;
}

__device__ glm::vec3 BRDF::fresnelSchlick(const glm::vec3& F0, float cosTheta)
{
	return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

__device__ float BRDF::distributionGGX(float NdotH, float roughness)
{
	const float a = roughness * roughness;
	const float a2 = a * a;
	float NdotH2 = NdotH * NdotH;

	const float num = a2;
	float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
	denom = glm::pi<float>() * denom * denom;

	return num / denom;
}

__device__ float BRDF::geometrySchlickGGX(float NdotV, float roughness)
{
	const float r = (roughness + 1.0f);
	const float k = (r * r) / 8.0f;

	float num = NdotV;
	float denom = NdotV * (1.0f - k) + k;
	return num / denom;
}

__device__ float BRDF::geometrySmith(float NdotV, float NdotL, float roughness)
{
	float ggx1 = geometrySchlickGGX(NdotV, roughness);
	float ggx2 = geometrySchlickGGX(NdotL, roughness);
	return ggx1 * ggx2;
}

__device__ glm::vec3 BRDF::sampleHemisphereCosineWeighted(const glm::vec3& N, uint32_t& seed)
{
	const float u1 = Random::Random::PcgFloat(seed);
	const float u2 = Random::Random::PcgFloat(seed);

	const float r = sqrt(u1);
	const float theta = 2.0f * glm::pi<float>() * u2;

	const float x = r * cos(theta);
	const float y = r * sin(theta);
	const float z = sqrt(1 - u1);

	glm::vec3 T, B; // Tangent and Bitangent
	if (fabs(N.x) > fabs(N.y))
		T = glm::vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z);
	else
		T = glm::vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);

	B = glm::cross(N, T);

	return x * T + y * B + z * N;
}

__device__ glm::vec3 BRDF::sampleGGX(const glm::vec3& N, float roughness, uint32_t& seed)
{
	const float u1 = Random::Random::PcgFloat(seed);
	const float u2 = Random::Random::PcgFloat(seed);

	const float a = roughness * roughness;

	const float cosTheta = sqrt((1.0f - u1) / (1.0f + (a * a - 1.0f) * u1));
	const float sinTheta = sqrt(1 - cosTheta * cosTheta);
	const float phi = 2.0f * glm::pi<float>() * u2;

	const auto H = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

	glm::vec3 T, B; // Tangent and Bitangent
	if (fabs(N.x) > fabs(N.y))
		T = glm::vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z);
	else
		T = glm::vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);

	B = glm::cross(N, T);

	return H.x * T + H.y * B + H.z * N;
}