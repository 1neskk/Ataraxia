#include "Application.h"
#include "main.h"
#include "Image.h"
#include "Renderer.h"
#include "Timer.h"

class CUDARayTracer final : public Layer
{
public:
    CUDARayTracer()
    {
        {
            std::vector<Sphere> sphereVec;

            Sphere s;
            s.center = { 0.0f, 0.0f, -3.0f };
            s.radius = 1.0f;
            s.id = 0;
            sphereVec.push_back(s);

            Sphere s1;
            s1.center = { -3.0f, 0.0f, -3.0f };
            s1.radius = 1.0f;
            s1.id = 1;
            sphereVec.push_back(s1);

            m_scene.setSpheres(sphereVec);
        }
    }
    
    virtual void onImGuiRender() override
    {
        ImGui::CreateContext();
        const auto& io = ImGui::GetIO();

        Style::theme();

        ImGui::Begin("Settings");
        if (ImGui::Button("Render"))
        {
            Render();
            std::cout << "Rendering..." << std::endl;
        }

        ImGui::Text("Last Render Time: %.3fms | (%.1f FPS)", m_lastRenderTime, io.Framerate);
        ImGui::End();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Viewport");
        m_viewportWidth = ImGui::GetContentRegionAvail().x;
        m_viewportHeight = ImGui::GetContentRegionAvail().y;

        if (const auto image = m_renderer.getImage())
            ImGui::Image((void*)(intptr_t)image->getTextureID(), { static_cast<float>(image->getWidth()),
            static_cast<float>(image->getHeight()) }, ImVec2(0, 1), ImVec2(1, 0));

        ImGui::End();
        ImGui::PopStyleVar();

        Render();
    }

    void Render()
    {
        Timer timer;

        m_renderer.onResize(m_viewportWidth, m_viewportHeight);
        m_renderer.Render(m_scene);

        m_lastRenderTime = timer.ElapsedMS();
    }

private:
    Scene m_scene;
    Renderer m_renderer;
    uint32_t m_viewportWidth = 0, m_viewportHeight = 0;
    float m_lastRenderTime = 0.0f;
};

Application* createApplication(int argc, char** argv)
{
    Specs spec;
    spec.name = "CUDARayTracer";

	auto app = new Application(spec);
    app->pushLayer<CUDARayTracer>();
	app->setMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
	         if (ImGui::MenuItem("Exit"))
	             app->close();
	         ImGui::EndMenu();
		}
	});
    return app;
}
