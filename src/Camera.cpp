#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Input/Input.h"

typedef Input::Input input;

Camera::Camera(float fov, float nearClip, float farClip)
	:m_fov(fov), m_nearClip(nearClip), m_farClip(farClip)
{
	m_direction = glm::vec3(0.0f, 0.0f, -1.0f);
	m_position = glm::vec3(0.0f, 0.0f, 3.0f);
}

bool Camera::onUpdate(float dt)
{
	bool moved = false;

	const glm::vec2 pos = input::GetMousePosition();
	const glm::vec2 delta = (pos - m_lastMousePos) * 0.002f;
	m_lastMousePos = pos;

	if (!input::IsMouseButtonPressed(Input::MouseButton::Right))
	{
		input::SetCursorState(Input::CursorState::Normal);
		return false;
	}

	input::SetCursorState(Input::CursorState::Locked);

	constexpr glm::vec3 up = { 0.0f, 1.0f, 0.0f };
	const glm::vec3 right = glm::cross(m_direction, up);

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

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		const float yaw = delta.x * getRotationSpeed();
		const float pitch = delta.y * getRotationSpeed();

		const glm::quat orientation = glm::normalize(glm::cross(glm::angleAxis(-pitch, right),
			glm::angleAxis(-yaw, glm::vec3(0.0f, 1.0f, 0.0f))));

		m_direction = glm::rotate(orientation, m_direction);
		moved = true;
	}
	if (moved)
	{
		std::cout << "Camera Position: " << m_position.x << ", " << m_position.y << ", " << m_position.z << std::endl;
	}
	return moved;
}
