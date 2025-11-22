#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <thread>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "Camera.h"
#include "Random.h"
#include "input/Input.h"

typedef Input::Input input;

Camera::Camera(float fov, float nearClip, float farClip)
    : m_fov(fov), m_nearClip(nearClip), m_farClip(farClip), m_width(1600),
      m_height(900) {
  m_direction = glm::vec3(0.0f, 0.0f, -1.0f);
  m_position = glm::vec3(0.0f, 0.0f, 3.0f);
  UpdateViewMatrix();
  UpdateProjectionMatrix();
}

Camera::Camera(float fov, float nearClip, float farClip, glm::vec3 position,
               glm::vec3 direction)
    : m_fov(fov), m_nearClip(nearClip), m_farClip(farClip),
      m_position(position), m_direction(direction), m_width(1600),
      m_height(900) {
  UpdateViewMatrix();
  UpdateProjectionMatrix();
}

bool Camera::onUpdate(float dt) {
  glm::vec2 mousePos = input::GetMousePosition();
  glm::vec2 mouseDelta = (mousePos - m_lastMousePos) * 0.002f;
  m_lastMousePos = mousePos;

  if (!input::IsMouseButtonPressed(Input::MouseButton::Right)) {
    input::SetCursorMode(Input::CursorMode::Normal);
    return false;
  }

  input::SetCursorMode(Input::CursorMode::Locked);
  bool moved = false;

  constexpr glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 right = glm::cross(m_direction, up);

  float speed = 5.0f;

  if (input::IsKeyPressed(Input::Key::W)) {
    m_position += m_direction * speed * dt;
    m_viewDirty = true;
    moved = true;
  } else if (input::IsKeyPressed(Input::Key::S)) {
    m_position -= m_direction * speed * dt;
    m_viewDirty = true;
    moved = true;
  }

  if (input::IsKeyPressed(Input::Key::A)) {
    m_position -= right * speed * dt;
    m_viewDirty = true;
    moved = true;
  } else if (input::IsKeyPressed(Input::Key::D)) {
    m_position += right * speed * dt;
    m_viewDirty = true;
    moved = true;
  }

  if (input::IsKeyPressed(Input::Key::Q)) {
    m_position -= up * speed * dt;
    m_viewDirty = true;
    moved = true;
  } else if (input::IsKeyPressed(Input::Key::E)) {
    m_position += up * speed * dt;
    m_viewDirty = true;
    moved = true;
  }

  if (mouseDelta.x != 0.0f || mouseDelta.y != 0.0f) {
    float yaw = mouseDelta.x * getRotationSpeed();
    float pitch = mouseDelta.y * getRotationSpeed();

    glm::quat orientation = glm::normalize(
        glm::cross(glm::angleAxis(-pitch, right),
                   glm::angleAxis(-yaw, glm::vec3(0.0f, 1.0f, 0.0f))));

    m_direction = glm::rotate(orientation, m_direction);
    m_viewDirty = true;
    moved = true;
  }
  if (m_viewDirty) {
    UpdateViewMatrix();
    UpdateViewMatrix();
  }
  return moved;
}

void Camera::Resize(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0) {
    std::cerr << "Error: Width or height cannot be zero.\n";
    return;
  }

  if (width == m_width && height == m_height)
    return;

  m_width = width;
  m_height = height;

  m_projectionDirty = true;
  UpdateProjectionMatrix();
  UpdateProjectionMatrix();
}

float Camera::getRotationSpeed() { return 0.3f; }

void Camera::UpdateProjectionMatrix() {
  if (m_projectionDirty) {
    float aspectRatio =
        static_cast<float>(m_width) / static_cast<float>(m_height);
    if (aspectRatio <= std::numeric_limits<float>::epsilon()) {
      std::cerr << "Error: invalid Aspect ratio\n";
      return;
    }

    m_projectionMatrix = glm::perspective(glm::radians(m_fov), aspectRatio,
                                          m_nearClip, m_farClip);
    m_inverseProjectionMatrix = glm::inverse(m_projectionMatrix);
    m_projectionDirty = false;
  }
}

void Camera::UpdateViewMatrix() {
  if (m_viewDirty) {
    m_viewMatrix = glm::lookAt(m_position, m_position + m_direction,
                               glm::vec3(0.0f, 1.0f, 0.0f));
    m_inverseViewMatrix = glm::inverse(m_viewMatrix);
    m_viewDirty = false;
  }
}

void Camera::allocateDevice(DeviceCamera &deviceCamera) const {
  deviceCamera.position = m_position;
  deviceCamera.direction = m_direction;
  deviceCamera.width = m_width;
  deviceCamera.height = m_height;
  deviceCamera.inverseViewMatrix = m_inverseViewMatrix;
  deviceCamera.inverseProjectionMatrix = m_inverseProjectionMatrix;
}

void Camera::freeDevice(DeviceCamera &deviceCamera) {}
