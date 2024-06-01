#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "imgui.h"
#include "vulkan/vulkan.h"

class Layer
{
public:
    virtual ~Layer() = default;

    virtual void onAttach() {}
    virtual void onDetach() {}

    virtual void onUpdate(float ts) {}
    virtual void onGuiRender() {}
};

void checkVkResult(VkResult result);

struct GLFWwindow;

struct Specs
{
    std::string name = "Vulkan Application";
    uint32_t width = 1280, height = 720;
};

class Application
{
public:
    Application(const Specs& specs = Specs());
    ~Application();

    static Application& get();

    void run();
    void setMenubarCallback(const std::function<void()>& callback) { m_menubarCallback = callback; }

    template<typename T>
    void pushLayer()
    {
        static_assert(std::is_base_of<Layer, T>::value, "T must derive from Layer");
        m_layers.emplace_back(std::make_shared<T>())->onAttach();
    }

    void pushLayer(const std::shared_ptr<Layer>& layer)
    {
        m_layers.emplace_back(layer);
        layer->onAttach();
    }

    void close();
    float getTime();

    GLFWwindow* getWindow() { return m_window; }

    static VkInstance getInstance();
    static VkPhysicalDevice getPhysicalDevice();
    static VkDevice getDevice();

    static VkCommandBuffer beginSingleTimeCommands();
    static VkCommandBuffer submitSingleTimeCommands(VkCommandBuffer commandBuffer);

    static VkCommandBuffer getCommandBuffer(bool begin);
    static void flushCommandBuffer(VkCommandBuffer commandBuffer);

    static void submitResourceFree(std::function<void()>&& func);

private:
    void init();
    void shutdown();

private:
    Specs m_specs;
    GLFWwindow* m_window = nullptr;
    bool m_running = true;

    float m_timeStep = 0.0f;
    float m_frameTime = 0.0f;
    float m_lastFrameTime = 0.0f;

    std::vector<std::shared_ptr<Layer>> m_layers;
    std::function<void()> m_menubarCallback;
};

Application* createApplication(int argc, char** argv);