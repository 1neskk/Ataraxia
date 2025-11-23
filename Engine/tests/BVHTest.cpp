#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include "BVH.h"
#include "SceneNode.h"
#include "Scene.h"

BVH g_bvh;

namespace {
    class BVHTest : public ::testing::Test {
    protected:
        std::vector<Sphere> spheres;
        size_t numSpheres = 1000;

        void SetUp() override {
            for (size_t i = 0; i < numSpheres; i++) {
                spheres.emplace_back(glm::vec3(i, i, i), 1.0f, 0);
            }
        }
    };

    void traverse(BVH& bvh) {
        uint32_t nodeIdx = 0;
        while (nodeIdx < bvh.getNodes().size()) {
            const auto& node = bvh.getNodes()[nodeIdx];
            if (node.count == 0) {
                nodeIdx = node.leftFirst;
            } else {
                for (uint32_t i = 0; i < node.count; i++) {
                    uint32_t sphereIdx = bvh.getSphereIndices()[node.leftFirst + i];
                    const Sphere& sphere = bvh.getSpheres()[sphereIdx];
                }
                nodeIdx++;
            }
        }
    }
}

TEST_F(BVHTest, Build) {
    g_bvh.build(spheres);

    ASSERT_EQ(g_bvh.getNodes().size(), numSpheres * 2);
    ASSERT_EQ(g_bvh.getSphereIndices().size(), numSpheres);
}

TEST_F(BVHTest, Traversal) {
    g_bvh.build(spheres);

    traverse(g_bvh);
    ASSERT_TRUE(true); // Assume it doesn't crash
}