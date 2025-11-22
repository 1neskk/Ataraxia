#pragma once

#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include "Camera.h"
#include "SceneNode.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Light {
  glm::vec3 position;
  glm::vec3 color;
  float intensity;

  Light() = default;
  Light(const glm::vec3 &pos, const glm::vec3 &col, float i)
      : position(pos), color(col), intensity(i) {}
};

struct Material {
  glm::vec3 albedo{1.0f};

  float roughness = 0.0f;
  float metallic = 0.0f;
  glm::vec3 F0{0.04f};

  int id = 0;

  Material() = default;
  Material(const glm::vec3 &albedo, float roughness, float metallic, int id)
      : albedo(albedo), roughness(roughness), metallic(metallic), id(id) {}
};

struct Settings {
  bool accumulation = true;
  bool skyLight = false;
  int maxBounces = 15;

  Settings() = default;
};

struct Scene {
  std::shared_ptr<SceneNode> rootNode;
  std::vector<Material> materials;
  std::vector<Light> lights;
  Settings settings;
  Camera camera;

  Scene() : rootNode(std::make_shared<SceneNode>("Scene")) {}

  Scene(const Scene &other) {
    rootNode = other.rootNode;
    materials = other.materials;
    lights = other.lights;
    settings = other.settings;
    camera = other.camera;
  }

  Scene(Scene &&other) noexcept = default;
  Scene &operator=(const Scene &other) = default;
  Scene &operator=(Scene &&other) noexcept = default;
};
