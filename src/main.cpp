#include "Application.h"
#include "main.h"
#include "Image.h"
#include "Renderer.h"
#include "Timer.h"

class PathTracingEngine final : public Layer
{
public:
    PathTracingEngine()
        : m_camera(45.0f, 0.1f, 100.0f)
    {
        Material& mat1 = m_scene.materials.emplace_back();
        mat1.albedo = { 0.5f, 0.0f, 1.0f };
        mat1.diffuse = 1.0f;

        Material& mat2 = m_scene.materials.emplace_back();
        mat2.albedo = { 1.0f, 0.0f, 0.0f };
        mat2.diffuse = 0.3f;

        Material& mat3 = m_scene.materials.emplace_back();
        mat3.albedo = { 0.8f, 0.5f, 0.2f };
        mat3.diffuse = 0.1f;
        mat3.emissionColor = mat3.albedo;
        mat3.emissionIntensity = 20.0f;

        Material& mat4 = m_scene.materials.emplace_back();
        mat4.albedo = { 1.0f, 1.0f, 1.0f };
        mat4.diffuse = { 0.0f };
        mat4.reflection = 0.8f;
        mat4.shininess = 0.1f;

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
            s.center = { 32.4f, 3.8f, -32.1f };
            s.radius = 20.3f;
            s.id = 2;
            m_scene.spheres.push_back(s);
        }

        {
            Sphere s;
            s.center = { -2.0f, 0.0f, 2.0f };
            s.radius = 1.0f;
            s.id = 3;
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
        ImGui::CreateContext();
        const auto& io = ImGui::GetIO();

        Style::theme();

        ImGui::Begin("Settings");

        ImGui::Checkbox("Accumulation", &m_renderer.getSettings().accumulation);
        if (ImGui::Button("Reset Frame Index"))
            m_renderer.resetFrameIndex();

        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 26);
        ImGui::Text("Last Render Time: %.3fms | (%.1f FPS)", m_lastRenderTime, io.Framerate);
        ImGui::End();

		ImGui::Begin("Scene settings");
		for (size_t i = 0; i < m_scene.spheres.size(); i++)
        {
            ImGui::PushID(i);
			ImGui::Text("Sphere %d", i);

            if (ImGui::DragFloat3("Position", &m_scene.spheres[i].center[0], 0.01f))
				m_renderer.resetFrameIndex();
			if (ImGui::DragFloat("Radius", &m_scene.spheres[i].radius, 0.01f))
				m_renderer.resetFrameIndex();

            ImGui::DragInt("Material index", &m_scene.spheres[i].id, 0.5f, 0, static_cast<int>(m_scene.materials.size() - 1));

            ImGui::Separator();
			ImGui::PopID();
		}
        ImGui::End();

        ImGui::Begin("Material settings");
		for (size_t i = 0; i < m_scene.materials.size(); i++)
        {
	        ImGui::PushID(i);
			ImGui::Text("Material %d", i);

			ImGui::ColorEdit3("Albedo", reinterpret_cast<float*>(&m_scene.materials[i].albedo));
			ImGui::DragFloat("Diffuse", &m_scene.materials[i].diffuse, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("Specular", &m_scene.materials[i].specular, 0.01f, 0.0f, 1.0f);
			ImGui::DragFloat("Shininess", &m_scene.materials[i].shininess, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Reflectivity", &m_scene.materials[i].reflection, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Transparency", &m_scene.materials[i].transparency, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Index of refraction", &m_scene.materials[i].ior, 0.01f, 0.0f, 2.0f);
            ImGui::ColorEdit3("Emission Color", reinterpret_cast<float*>(&m_scene.materials[i].emissionColor));
			ImGui::DragFloat("Emission Intensity", &m_scene.materials[i].emissionIntensity, 0.05f, 0.0f, FLT_MAX);

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
    spec.name = "Path Tracing Engine";

	auto app = new Application(spec);
    app->pushLayer<PathTracingEngine>();
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
