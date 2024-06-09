#include "Application.h"
#include "main.h"
#include "Image.h"
#include "Renderer.h"
#include "Timer.h"

class CUDARayTracer final : public Layer
{
public:
    CUDARayTracer()
        : m_camera(45.0f, 0.1f, 100.0f)
    {
		Material& m1 = m_scene.materials.emplace_back();
		m1.albedo = { 1.0f, 1.0f, 1.0f };
		m1.diffuse = 0.3f;

		Material& m2 = m_scene.materials.emplace_back();
		m2.albedo = { 1.0f, 0.0f, 0.0f };
        m2.diffuse = 1.0f;

        {
            Sphere s;
            s.center = { 0.0f, 0.0f, -3.0f };
            s.radius = 1.0f;
            s.id = 0;
            m_scene.spheres.emplace_back(s);
        }
    }

    virtual void onUpdate(float ts) override
    {
        m_camera.onUpdate(ts);
    }
    
    virtual void onGuiRender() override
    {
        ImGui::CreateContext();
        const auto& io = ImGui::GetIO();

        Style::theme();

        ImGui::Begin("Settings");
        if (ImGui::Button("Render"))
        {
            Render();
            std::cout << "Rendering...\n";
        }

        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 26);
        ImGui::Text("Last Render Time: %.3fms | (%.1f FPS)", m_lastRenderTime, io.Framerate);
        ImGui::End();

		ImGui::Begin("Scene settings");
		for (size_t i = 0; i < m_scene.spheres.size(); i++)
        {
            ImGui::PushID(i);
			ImGui::Text("Sphere %d", i);

            ImGui::DragFloat3("Position", &m_scene.spheres[i].center[0], 0.01f);
			ImGui::DragFloat("Radius", &m_scene.spheres[i].radius, 0.01f);

            ImGui::DragInt("Material index", &m_scene.spheres[i].id, 0.5f, 0, static_cast<int>(m_scene.materials.size() - 1));

            ImGui::Separator();
			ImGui::PopID();
		}
        ImGui::End();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Viewport");
        m_viewportWidth = ImGui::GetContentRegionAvail().x;
        m_viewportHeight = ImGui::GetContentRegionAvail().y;

		if (const auto image = m_renderer.getImage())
            ImGui::Image(image->getDescriptorSet(), { static_cast<float>(image->getWidth()), static_cast<float>(image->getHeight()) },
                ImVec2(0, 1), ImVec2(1, 0));

        ImGui::End();
        ImGui::PopStyleVar();

        Render();
    }

    void Render()
    {
        Timer timer;

        m_renderer.onResize(m_viewportWidth, m_viewportHeight);
        m_camera.Resize(m_viewportWidth, m_viewportHeight);
        m_renderer.Render(m_camera, m_scene);

        m_lastRenderTime = timer.ElapsedMS();
    }

private:
    Scene m_scene;
    Renderer m_renderer;
	Camera m_camera;

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
