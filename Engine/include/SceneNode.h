#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct Sphere
{
    glm::vec3 center;
    float radius;
    int id = 0;

    Sphere() = default;
    Sphere(const glm::vec3& c, float r, int materialId)
        : center(c), radius(r), id(materialId) {
    }
};

class SceneNode
{
public:
    SceneNode();
    SceneNode(const std::string& name);
    ~SceneNode() = default;

    void setPosition(const glm::vec3& position) { m_position = position; m_transformDirty = true; }
    void setRotation(const glm::quat& rotation) { m_rotation = rotation; m_transformDirty = true; }
    void setScale(const glm::vec3& scale) { m_scale = scale; m_transformDirty = true; }

    const glm::vec3& getPosition() const { return m_position; }
    const glm::quat& getRotation() const { return m_rotation; }
    const glm::vec3& getScale() const { return m_scale; }

    void addChild(std::shared_ptr<SceneNode> child);
    void removeChild(std::shared_ptr<SceneNode> child);
    const std::vector<std::shared_ptr<SceneNode>>& getChildren() const { return m_children; }

    void addSphere(const Sphere& sphere);
    void removeSphere(int sphereIndex);
    const std::vector<Sphere>& getSpheres() const { return m_spheres; }

    void updateGlobalTransform(const glm::mat4& parentTransform = glm::mat4(1.0f));
    const glm::mat4& getGlobalTransform() const { return m_globalTransform; }

    const std::string& getName() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

private:
    std::string m_name;
    
    glm::vec3 m_position;
    glm::quat m_rotation;
    glm::vec3 m_scale;

    glm::mat4 m_localTransform;
    glm::mat4 m_globalTransform;

    std::vector<std::shared_ptr<SceneNode>> m_children;
    std::vector<Sphere> m_spheres;

    bool m_transformDirty;
};