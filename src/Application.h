#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "imgui.h"

class Layer
{
public:
    virtual ~Layer() = default;

    virtual void onAttach() {}
    virtual void onDetach() {}
    virtual void onUpdate(float ts) {}
    virtual void onImGuiRender() {}
};

struct GLFWwindow;

struct Specs
{
    std::string name = "Application";
    const uint32_t width = 1280, height = 720;
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

    GLFWwindow* getWindow() const { return m_window; }

    
    static void submitResourceFree(std::function<void()>&& func);

private:
    void init();
    void shutdown();

private:
    Specs m_specs;
    bool m_running = true;
    GLFWwindow* m_window = nullptr;

    float m_lastFrameTime = 0.0f;
    float m_frameTime = 0.0f;
    float m_timeStep = 0.0f;

    std::vector<std::shared_ptr<Layer>> m_layers;
    std::function<void()> m_menubarCallback;
};

Application* createApplication(int argc, char** argv);
