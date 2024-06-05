#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "input/Input.h"

typedef Input::Input input;

Camera::Camera(float fov, float nearClip, float farClip)
	: m_fov(fov), m_nearClip(nearClip), m_farClip(farClip)
{
	m_direction = glm::vec3(0.0f, 0.0f, -1.0f);
	m_position = glm::vec3(0.0f, 0.0f, 3.0f);
}

bool Camera::onUpdate(float dt)
{
	glm::vec2 mousePos = input::GetMousePosition();
	glm::vec2 mouseDelta = (mousePos - m_lastMousePos) * 0.002f;
	m_lastMousePos = mousePos;

	if (!input::IsMouseButtonPressed(Input::MouseButton::Right))
	{
		input::SetCursorMode(Input::CursorMode::Normal);
		return false;
	}

	input::SetCursorMode(Input::CursorMode::Locked);
	bool moved = false;

	constexpr glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 right = glm::cross(m_direction, up);

	float speed = 5.0f;

	if (input::IsKeyPressed(Input::Key::W))
	{
		m_position += m_direction * speed * dt;
		moved = true;
	}
	else if (input::IsKeyPressed(Input::Key::S))
	{
		m_position -= m_direction * speed * dt;
		moved = true;
	}

	if (input::IsKeyPressed(Input::Key::A))
	{
		m_position -= right * speed * dt;
		moved = true;
	}
	else if (input::IsKeyPressed(Input::Key::D))
	{
		m_position += right * speed * dt;
		moved = true;
	}

	if (input::IsKeyPressed(Input::Key::Q))
	{
		m_position -= up * speed * dt;
		moved = true;
	}
	else if (input::IsKeyPressed(Input::Key::E))
	{
		m_position += up * speed * dt;
		moved = true;
	}

	if (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f)
	{
		float yaw = mouseDelta.x * getRotationSpeed();
		float pitch = mouseDelta.y * getRotationSpeed();

		glm::quat orientation = glm::normalize(glm::cross(glm::angleAxis(-pitch, right), glm::angleAxis(-yaw, glm::vec3(0.0f, 1.0f, 0.0f))));

		m_direction = glm::rotate(orientation, m_direction);
		moved = true;
	}
	if (moved)
	{
		UpdateViewMatrix();
		UpdateRayDirection();
	}
	return moved;
}

void Camera::Resize(uint32_t width, uint32_t height)
{
	if (width == m_width && height == m_height)
		return;

	m_width = width;
	m_height = height;

	UpdateProjectionMatrix();
	UpdateRayDirection();
}

float Camera::getRotationSpeed()
{
	return 0.3f;
}

void Camera::UpdateProjectionMatrix()
{
	m_projectionMatrix = glm::perspective(glm::radians(m_fov), static_cast<float>(m_width) / static_cast<float>(m_height), m_nearClip, m_farClip);
	m_inverseProjectionMatrix = glm::inverse(m_projectionMatrix);
}

void Camera::UpdateViewMatrix()
{
	m_viewMatrix = glm::lookAt(m_position, m_position + m_direction, glm::vec3(0.0f, 1.0f, 0.0f));
	m_inverseViewMatrix = glm::inverse(m_viewMatrix);
}

void Camera::UpdateRayDirection()
{
	m_rayDirection.resize(m_width * m_height);

	for (uint32_t y = 0; y < m_height; y++)
	{
		for (uint32_t x = 0; x < m_width; x++)
		{
			glm::vec2 coord = { static_cast<float>(x) / static_cast<float>(m_width),
				static_cast<float>(y) / static_cast<float>(m_height) };
			coord = coord * 2.0f - 1.0f;

			glm::vec4 target = m_inverseProjectionMatrix * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
			glm::vec3 rayDir = glm::vec3(m_inverseViewMatrix * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f));
			m_rayDirection[x + y * m_width] = rayDir;
		}
	}
}