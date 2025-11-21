#include <gtest/gtest.h>
#include "BVH.h"

TEST(BVHTest, Construction) {
    BVH bvh;
    EXPECT_EQ(bvh.getNodes(), nullptr);
}