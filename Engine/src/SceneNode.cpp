#include "SceneNode.h"

SceneNode::SceneNode()
    : m_name("Untitled"), m_position(0.0f), m_rotation(glm::quat()), m_scale(1.0f),
      m_localTransform(1.0f), m_globalTransform(1.0f), m_transformDirty(true)
{
}

SceneNode::SceneNode(const std::string& name)
    : m_name(name), m_position(0.0f), m_rotation(glm::quat()), m_scale(1.0f),
      m_localTransform(1.0f), m_globalTransform(1.0f), m_transformDirty(true)
{
}

void SceneNode::addChild(std::shared_ptr<SceneNode> child)
{
    m_children.emplace_back(child);
}

void SceneNode::removeChild(std::shared_ptr<SceneNode> child)
{
    m_children.erase(std::remove(m_children.begin(), m_children.end(), child), m_children.end());
}

void SceneNode::addSphere(const Sphere& sphere)
{
    m_spheres.emplace_back(sphere);
}

void SceneNode::removeSphere(int sphereIndex)
{
    if (sphereIndex >= 0 && sphereIndex < static_cast<int>(m_spheres.size()))
    {
        m_spheres.erase(m_spheres.begin() + sphereIndex);
    }
}

void SceneNode::updateGlobalTransform(const glm::mat4& parentTransform)
{
    if (m_transformDirty)
    {
        m_localTransform = glm::translate(glm::mat4(1.0f), m_position) * glm::mat4_cast(m_rotation) * glm::scale(glm::mat4(1.0f), m_scale);
        m_globalTransform = parentTransform * m_localTransform;
        m_transformDirty = false;
    }
    else
    {
        m_globalTransform = parentTransform * m_localTransform;
    }

    for (const auto& child : m_children)
    {
        child->updateGlobalTransform(m_globalTransform);
    }
}
