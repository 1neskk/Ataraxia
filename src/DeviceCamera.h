#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct DeviceCamera
{
	glm::vec3 position, direction;

	glm::vec3* rayDirection;
	uint32_t width, height;
};