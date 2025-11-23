#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "SceneNode.h"

struct BVHNode {
    glm::vec3 aabbMin;
    uint32_t leftFirst;  // If leaf: index to first primitive. If internal:
                         // index to left child.
    glm::vec3 aabbMax;
    uint32_t count;  // If > 0, it's a leaf and this is primitive count. If 0,
                     // it's internal.
};

class BVH {
   public:
    BVH() = default;
    void build(const std::vector<Sphere>& spheres);

    const std::vector<BVHNode>& getNodes() const { return m_nodes; }
    const std::vector<int>& getSphereIndices() const { return m_sphereIndices; }

   private:
    void updateNodeBounds(uint32_t nodeIdx);
    void subdivide(uint32_t nodeIdx);

   private:
    std::vector<BVHNode> m_nodes;
    std::vector<int> m_sphereIndices;
    const std::vector<Sphere>* m_spheres = nullptr;
    uint32_t m_nodesUsed = 0;
};
