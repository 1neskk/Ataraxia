#pragma once

#include <glm/glm.hpp>

struct DeviceCamera
{
	glm::vec3 position, direction;

	glm::vec3* rayDirection;
	uint32_t width, height;
};