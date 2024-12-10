#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct DeviceCamera
{
	glm::vec3 position, direction;

	glm::vec3* rayDirection;
	uint32_t width, height;

	//float shutterCloseTime = 1.0f;
	//float shutterOpenTime = 0.0f;

	//float lensRadius = 0.0f;

	//glm::vec3 right = { 1.0f, 0.0f, 0.0f };
	//glm::vec3 up = { 0.0f, 1.0f, 0.0f };

	//float focalDistance = 1.0f;

	//float velocity = 5.0f;
};