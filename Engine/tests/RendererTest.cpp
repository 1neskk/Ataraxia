#include <gtest/gtest.h>
#include "Renderer.h"
#include "Scene.h"
#include "Camera.h"
#include <cuda_runtime.h>

bool g_bRunning = true;

namespace {
    bool HasCudaDevice() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return error == cudaSuccess && deviceCount > 0;
    }

    class RendererTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!HasCudaDevice()) {
                GTEST_SKIP() << "No CUDA device available, skipping Renderer tests that require GPU.";
            }
        }
    };

    TEST(RendererSafeTest, Construction) {
        Renderer renderer;
        EXPECT_EQ(renderer.getImage(), nullptr);
    }

    TEST(RendererSafeTest, Settings) {
        Renderer renderer;
        Settings settings;
        settings.maxBounces = 10;
        settings.skyLight = true;
        
        renderer.setSettings(settings);
        
        Settings retrieved = renderer.getSettings();
        EXPECT_EQ(retrieved.maxBounces, 10);
        EXPECT_EQ(retrieved.skyLight, true);
    }

    TEST_F(RendererTest, OnResize) {
        Renderer renderer;
        renderer.setHeadless(true);
        uint32_t width = 800;
        uint32_t height = 600;
        
        renderer.onResize(width, height);
        
        auto image = renderer.getImage();
        EXPECT_EQ(image, nullptr);
    }

    TEST_F(RendererTest, Render) {
        Renderer renderer;
        renderer.setHeadless(true);
        uint32_t width = 100;
        uint32_t height = 100;
        renderer.onResize(width, height);
        
        Scene scene;
        scene.camera.Resize(width, height);
        
        renderer.Render(scene.camera, scene);
        
        auto image = renderer.getImage();
        EXPECT_EQ(image, nullptr);
    }

    TEST_F(RendererTest, ResetFrameIndex) {
        Renderer renderer;
        renderer.resetFrameIndex();
        SUCCEED();
    }
}
