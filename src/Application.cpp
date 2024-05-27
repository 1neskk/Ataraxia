#include "Application.h"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl2.h"
#include <stdio.h>
#include <stdlib.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include <iostream>

#include "imgui/Roboto-Regular.embed"

extern bool g_bRunning;

static std::vector<std::vector<std::function<void()>>> s_ResourceFreeQueue;
static uint32_t s_CurrentFrameIndex = 0;
static Application* s_Instance = nullptr;

static void glfwErrorCallback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

Application::Application(const Specs& specs)
    : m_specs(specs)
{
    s_Instance = this;
    init();
}

Application::~Application()
{
    shutdown();
    s_Instance = nullptr;
}

Application& Application::get()
{
    return *s_Instance;
}

void Application::init()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(glfwErrorCallback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    m_window = glfwCreateWindow(m_specs.width, m_specs.height, m_specs.name.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

#ifndef _GLFW_WAYLAND
    glfwSetWindowPos(m_window, 100, 100);
#endif

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL2_Init();

    ImFontConfig fontConfig;
    fontConfig.FontDataOwnedByAtlas = false;
    ImFont* robotoFont = io.Fonts->AddFontFromMemoryTTF((void*)g_RobotoRegular, sizeof(g_RobotoRegular), 18.0f, &fontConfig);
    io.FontDefault = robotoFont;

    // glEnable(GL_DEBUG_OUTPUT);
    // glDebugMessageCallback([](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
    //     {
    //         if (severity != GL_DEBUG_SEVERITY_NOTIFICATION)
    //         {
    //             std::cerr << message << std::endl;
    //         }
    //     }, nullptr);

}

void Application::shutdown()
{
    for (auto& layer : m_layers)
        layer->onDetach();

    m_layers.clear();

    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();

    for (auto& queue : s_ResourceFreeQueue)
        queue.clear();

    g_bRunning = false;
}

void Application::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        float time = getTime();
        m_frameTime = time - m_lastFrameTime;
        m_lastFrameTime = time;
        m_timeStep = std::min(m_frameTime, 0.0333f);

        glClear(GL_COLOR_BUFFER_BIT);

        for (auto& layer : m_layers)
            layer->onUpdate();

        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        for (auto& layer : m_layers)
            layer->onImGuiRender();

        ImGui::Render();
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void Application::close()
{
    m_running = false;
}

float Application::getTime()
{
    return static_cast<float>(glfwGetTime());
}

void Application::submitResourceFree(std::function<void()>&& func)
{
    s_ResourceFreeQueue[s_CurrentFrameIndex].emplace_back(std::move(func));
}
