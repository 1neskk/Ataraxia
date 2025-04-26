#pragma once

#include <vector>
#include "DeviceMemory.h"

struct DeviceCamera
{
	glm::vec3 position, direction;
	uint32_t width, height;

	glm::vec3* rayDirection;
};

class Camera
{
public:
	Camera() = default;
	Camera(float fov, float nearClip, float farClip);
	Camera(float fov, float nearClip, float farClip, glm::vec3 position, glm::vec3 direction);

	Camera& operator=(const Camera& other)
	{
		m_projectionMatrix = other.m_projectionMatrix;
		m_viewMatrix = other.m_viewMatrix;
		m_inverseProjectionMatrix = other.m_inverseProjectionMatrix;
		m_inverseViewMatrix = other.m_inverseViewMatrix;
		m_position = other.m_position;
		m_direction = other.m_direction;
		m_rayDirection = other.m_rayDirection;
		m_fov = other.m_fov;
		m_nearClip = other.m_nearClip;
		m_farClip = other.m_farClip;
		m_lastMousePos = other.m_lastMousePos;
		m_width = other.m_width;
		m_height = other.m_height;
		m_viewDirty = other.m_viewDirty;
		m_projectionDirty = other.m_projectionDirty;
		return *this;
	}

	bool onUpdate(float dt);
	void Resize(uint32_t width, uint32_t height);

	void allocateDevice(DeviceCamera& deviceCamera) const;
	static void freeDevice(DeviceCamera& deviceCamera);

	const glm::mat4& getViewMatrix() const { return m_viewMatrix; }
	const glm::mat4& getProjectionMatrix() const { return m_projectionMatrix; }
	const glm::mat4& getInverseViewMatrix() const { return m_inverseViewMatrix; }
	const glm::mat4& getInverseProjectionMatrix() const { return m_inverseProjectionMatrix; }

	const glm::vec3& getPosition() const { return m_position; }
	const glm::vec3& getDirection() const { return m_direction; }
	const float& getFov() const { return m_fov; }

	void setPosition(const glm::vec3& position) { m_position = position; m_viewDirty = true; }
	void setDirection(const glm::vec3& direction) { m_direction = direction; m_viewDirty = true; }
	void setFov(float fov) { m_fov = fov; m_projectionDirty = true; }

	const std::vector<glm::vec3>& getRayDirection() const { return m_rayDirection; }
	static float getRotationSpeed();

private:
	void UpdateProjectionMatrix();
	void UpdateViewMatrix();
	void UpdateRayDirection();
private:
	glm::mat4 m_projectionMatrix{ 1.0f };
	glm::mat4 m_viewMatrix{ 1.0f };
	glm::mat4 m_inverseProjectionMatrix{ 1.0f };
	glm::mat4 m_inverseViewMatrix{ 1.0f };

	glm::vec3 m_position{ 0.0f };
	glm::vec3 m_direction{ 0.0f };

	std::vector<glm::vec3> m_rayDirection;
	CudaBuffer<glm::vec3> m_deviceRayDirBuffer;

	float m_fov = 45.0f;
	float m_nearClip = 0.1f;
	float m_farClip = 100.0f;

	glm::vec2 m_lastMousePos{ 0.0f };
	uint32_t m_width = 0, m_height = 0;

	bool m_viewDirty = true;
	bool m_projectionDirty = true;
};
