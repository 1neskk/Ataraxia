#include "Application.h"
#include "main.h"
#include "Image.h"
#include "Renderer.h"
#include "Timer.h"

class Ataraxia final : public Layer
{
public:
    Ataraxia()
        : m_camera(45.0f, 0.1f, 100.0f)
    {
        ImGui::CreateContext();

        Material& mat1 = m_scene.materials.emplace_back();
        mat1.albedo = { 1.022f, 0.782f, 0.344f };
        mat1.roughness = 1.0f;

        Material& mat2 = m_scene.materials.emplace_back();
        mat2.albedo = { 1.0f, 0.0f, 0.0f };
        mat2.roughness = 0.3f;

        Material& mat4 = m_scene.materials.emplace_back();
		mat4.albedo = { 0.972f, 0.960f, 0.915f };
		mat4.roughness = 0.25f;
		mat4.metallic = 1.0f;
		mat4.F0 = { 0.96f, 0.96f, 0.97f };

        {
            Light l;
			l.intensity = 1.0f;
			l.color = { 1.0f, 1.0f, 1.0f };
			l.position = { 10.0f, 10.0f, 0.0f };
			m_scene.lights.push_back(l);
        }

        {
            Sphere s;
            s.center = { 0.0f, 0.0f, 0.0f };
            s.radius = 1.0f;
            s.id = 0;
            m_scene.spheres.push_back(s);
        }

        {
            Sphere s;
            s.center = { 0.0f, -101.0f, 0.0f };
            s.radius = 100.0f;
            s.id = 1;
            m_scene.spheres.push_back(s);
        }

        {
            Sphere s;
            s.center = { -2.0f, 0.0f, 2.0f };
            s.radius = 1.0f;
            s.id = 2;
            m_scene.spheres.push_back(s);
        }
    }

    virtual void onUpdate(const float ts) override
    {
        if (m_camera.onUpdate(ts))
            m_renderer.resetFrameIndex();
    }
    
    virtual void onGuiRender() override
    {
        const auto& io = ImGui::GetIO();

        Style::theme();

        ImGui::Begin("Settings");

        ImGui::Checkbox("Accumulation", const_cast<bool*>(&m_renderer.getSettings().accumulation));
        if (ImGui::Button("Reset Frame Index"))
            m_renderer.resetFrameIndex();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
        ImGui::Checkbox("Sky Light", const_cast<bool*>(&m_renderer.getSettings().skyLight));
        ImGui::DragInt("Max Bounces", const_cast<int*>(&m_renderer.getSettings().maxBounces), 1, 1, 500);

        ImGui::Separator();
        ImGui::Text("Last Render Time: %.3fms | (%.1f FPS)", m_lastRenderTime, io.Framerate);
        ImGui::End();

		ImGui::Begin("Scene settings");
		for (size_t i = 0; i < m_scene.spheres.size(); i++)
        {
            ImGui::PushID(static_cast<int32_t>(i));
			ImGui::Text("Sphere %d", static_cast<int32_t>(i) + 1);

            if (ImGui::DragFloat3("Position", &m_scene.spheres[i].center[0], 0.01f))
				m_renderer.resetFrameIndex();
			if (ImGui::DragFloat("Radius", &m_scene.spheres[i].radius, 0.01f))
				m_renderer.resetFrameIndex();

            if (!m_scene.materials.empty())
                ImGui::DragInt("Material index", &m_scene.spheres[i].id, 0.5f, 0, static_cast<int>(m_scene.materials.size() - 1));

            ImGui::Separator();
			ImGui::PopID();
		}
        ImGui::End();

        ImGui::Begin("Material settings");
		for (size_t i = 0; i < m_scene.materials.size(); i++)
        {
	        ImGui::PushID(static_cast<int32_t>(i));
			ImGui::Text("Material %d", static_cast<int32_t>(i) + 1);

			ImGui::ColorEdit3("Albedo", reinterpret_cast<float*>(&m_scene.materials[i].albedo));
			ImGui::DragFloat("Roughness", &m_scene.materials[i].roughness, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("Metallic", &m_scene.materials[i].metallic, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat3("F0", &m_scene.materials[i].F0[0], 0.01f, 0.0f, 1.0f);

			ImGui::ColorEdit3("Emission Color", reinterpret_cast<float*>(&m_scene.materials[i].emissionColor));
			ImGui::DragFloat("Emission Intensity", &m_scene.materials[i].emissionIntensity, 0.01f, 0.0f, FLT_MAX);

			ImGui::Separator();
			ImGui::PopID();
		}
		ImGui::End();

		ImGui::Begin("Light settings");
        for (size_t i = 0; i < m_scene.lights.size(); i++)
        {
			ImGui::PushID(static_cast<int32_t>(i));
			ImGui::Text("Light %d", static_cast<int32_t>(i) + 1);

			ImGui::DragFloat3("Position", &m_scene.lights[i].position[0], 0.01f);
			//ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&m_scene.lights[i].color));
			ImGui::DragFloat("Intensity", &m_scene.lights[i].intensity, 0.01f, 0.0f, FLT_MAX);

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
    spec.name = "Ataraxia";

	auto app = new Application(spec);
    app->pushLayer<Ataraxia>();
	app->setMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
	         if (ImGui::MenuItem("Exit"))
	             app->close();
			 if (ImGui::MenuItem("Fullscreen", "F11"))
				app->toggleFullscreen();
	         ImGui::EndMenu();
		}
	});
    return app;
}
