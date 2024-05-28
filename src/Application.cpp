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
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    m_window = glfwCreateWindow(m_specs.width, m_specs.height, m_specs.name.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    
    if (!getenv("WAYLAND_DISPLAY"))
    {
        glfwSetWindowPos(m_window, 100, 100);
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    int w, h;
    glfwGetFramebufferSize(m_window, &w, &h);

    s_ResourceFreeQueue.resize(2);

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
    m_running = true;
    
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO& io = ImGui::GetIO();

    while (!glfwWindowShouldClose(m_window) && m_running)
    {
        glfwPollEvents();

        for (auto& layer : m_layers)
            layer->onUpdate(m_timeStep);

        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
            if (m_menubarCallback)
                window_flags |= ImGuiWindowFlags_MenuBar;

            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->Pos);
            ImGui::SetNextWindowSize(viewport->Size);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

            if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
                window_flags |= ImGuiWindowFlags_NoBackground;

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("DockSpace Demo", nullptr, window_flags);
            ImGui::PopStyleVar();

            ImGui::PopStyleVar(2);

            if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
            {
                ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
                ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
            }

            if (m_menubarCallback)
            {
                if (ImGui::BeginMenuBar())
                {
                    m_menubarCallback();
                    ImGui::EndMenuBar();
                }
            }

            for (auto& layer : m_layers)
                layer->onImGuiRender();

            ImGui::End();
        }

        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        const bool isMinimized = draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f;
        if (!isMinimized)
        {
            glViewport(0, 0, static_cast<int>(draw_data->DisplaySize.x), static_cast<int>(draw_data->DisplaySize.y));
            glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL2_RenderDrawData(draw_data);
        }

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(m_window);
        }

        glfwSwapBuffers(m_window);

        float time = getTime();
        m_frameTime = time - m_lastFrameTime;
        m_timeStep = std::min(m_frameTime, 0.0333f);
        m_lastFrameTime = time;
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
