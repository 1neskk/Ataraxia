#include "BVH.h"
#include <algorithm>

void BVH::build(const std::vector<Sphere> &spheres) {
  m_spheres = &spheres;
  m_nodes.resize(spheres.size() * 2);
  m_sphereIndices.resize(spheres.size());
  for (size_t i = 0; i < spheres.size(); i++)
    m_sphereIndices[i] = i;

  BVHNode &root = m_nodes[0];
  root.leftFirst = 0;
  root.count = spheres.size();
  m_nodesUsed = 1;

  updateNodeBounds(0);
  subdivide(0);
}

void BVH::updateNodeBounds(uint32_t nodeIdx) {
  BVHNode &node = m_nodes[nodeIdx];
  node.aabbMin = glm::vec3(1e30f);
  node.aabbMax = glm::vec3(-1e30f);

  for (uint32_t i = 0; i < node.count; i++) {
    uint32_t sphereIdx = m_sphereIndices[node.leftFirst + i];
    const Sphere &sphere = (*m_spheres)[sphereIdx];

    node.aabbMin =
        glm::min(node.aabbMin, sphere.center - glm::vec3(sphere.radius));
    node.aabbMax =
        glm::max(node.aabbMax, sphere.center + glm::vec3(sphere.radius));
  }
}

void BVH::subdivide(uint32_t nodeIdx) {
  BVHNode &node = m_nodes[nodeIdx];

  if (node.count <= 2)
    return;

  glm::vec3 extent = node.aabbMax - node.aabbMin;
  int axis = 0;
  if (extent.y > extent.x)
    axis = 1;
  if (extent.z > extent[axis])
    axis = 2;

  float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

  int i = node.leftFirst;
  int j = i + node.count - 1;

  while (i <= j) {
    if ((*m_spheres)[m_sphereIndices[i]].center[axis] < splitPos)
      i++;
    else
      std::swap(m_sphereIndices[i], m_sphereIndices[j--]);
  }

  int leftCount = i - node.leftFirst;
  if (leftCount == 0 || leftCount == node.count)
    return;

  int leftChildIdx = m_nodesUsed++;
  int rightChildIdx = m_nodesUsed++;

  m_nodes[leftChildIdx].leftFirst = node.leftFirst;
  m_nodes[leftChildIdx].count = leftCount;
  m_nodes[rightChildIdx].leftFirst = i;
  m_nodes[rightChildIdx].count = node.count - leftCount;

  node.leftFirst = leftChildIdx;
  node.count = 0;

  updateNodeBounds(leftChildIdx);
  updateNodeBounds(rightChildIdx);

  subdivide(leftChildIdx);
  subdivide(rightChildIdx);
}
