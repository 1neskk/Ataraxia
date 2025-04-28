#include "Application.h"
#include "main.h"
#include "Image.h"
#include "Renderer.h"
#include "Utils.h"
#include "Timer.h"

class Ataraxia final : public Layer
{
public:
    Ataraxia()
        : m_camera(45.0f, 0.1f, 100.0f)
    {
        m_scene.camera = m_camera;
        m_scene.settings = m_renderer.getSettings();

        ImGui::CreateContext();

        initializeScene();
    }

    virtual void onUpdate(const float ts) override
    {
        if (m_camera.onUpdate(ts))
        {
            m_renderer.resetFrameIndex();
            m_scene.camera = m_camera;
            m_scene.settings = m_renderer.getSettings();
        }

        m_scene.rootNode->updateGlobalTransform();
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
        ImGui::DragInt("Ray Depth", const_cast<int*>(&m_renderer.getSettings().maxBounces), 1, 1, 500);

        if (ImGui::DragFloat("FOV", const_cast<float*>(&m_camera.getFov()), 1.0f, 179.0f))
        {
            m_camera = Camera(m_camera.getFov(), 0.1f, 100.0f, m_camera.getPosition(), m_camera.getDirection());
            m_scene.camera = m_camera;
            m_renderer.resetFrameIndex();
        }
        if (ImGui::Button("Reset Camera"))
        {
            m_camera = Camera(45.0f, 0.1f, 100.0f);
            m_scene.camera = m_camera;
            m_renderer.resetFrameIndex();
        }

        ImGui::Separator();
        ImGui::Text("Last Render Time: %.3fms | (%.1f FPS)", m_lastRenderTime, io.Framerate);

        ImGui::End();

        ImGui::Begin("Hierarchy");
        static std::function<void(const std::shared_ptr<SceneNode>&)> drawNode = [&](const std::shared_ptr<SceneNode>& node)
        {
            ImGui::PushID(node->getName().c_str());

            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanFullWidth;
            if (ImGui::TreeNodeEx(node->getName().c_str(), flags))
            {
                ImGui::Indent();

                ImGui::Text("Transform");
                ImGui::Separator();

                if (ImGui::DragFloat3("Position", const_cast<float*>(&node->getPosition()[0]), 0.01f))
                {
                    node->setPosition(node->getPosition());
                    m_renderer.resetFrameIndex();
                }
                if (ImGui::DragFloat3("Rotation", const_cast<float*>(&node->getRotation()[0]), 0.01f))
                {
                    node->setRotation(node->getRotation());
                    m_renderer.resetFrameIndex();
                }
                if (ImGui::DragFloat3("Scale", const_cast<float*>(&node->getScale()[0]), 0.01f, 0.0f, FLT_MAX))
                {
                    node->setScale(node->getScale());
                    m_renderer.resetFrameIndex();
                }

                ImGui::Spacing();
                if (ImGui::Button("Remove Node"))
                {
                    m_scene.rootNode->removeChild(node);
                    m_renderer.resetFrameIndex();
                }

                ImGui::Spacing();

                if (!node->getSpheres().empty())
                {
                    ImGui::Text("Spheres");
                    ImGui::Separator();
                    for (size_t i = 0; i < node->getSpheres().size(); i++)
                    {
                        ImGui::PushID(static_cast<int32_t>(i));
                        std::string sphereLabel = "Sphere " + std::to_string(i + 1);
                        if (ImGui::CollapsingHeader(sphereLabel.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
                        {
                            if (ImGui::DragFloat3("Center", const_cast<float*>(&node->getSpheres()[i].center[0]), 0.01f))
                                m_renderer.resetFrameIndex();
                            if (ImGui::DragFloat("Radius", const_cast<float*>(&node->getSpheres()[i].radius), 0.01f))
                                m_renderer.resetFrameIndex();
                            if (ImGui::Combo("Material", const_cast<int*>(&node->getSpheres()[i].id), "Material 1\0Material 2\0Material 3\0\0"))
                                m_renderer.resetFrameIndex();
                        }
                        ImGui::PopID();
                    }
                }

                for (const auto& child : node->getChildren())
                {
                    drawNode(child);
                }
                ImGui::Unindent();
                ImGui::TreePop();
            }
            ImGui::PopID();
        };

        drawNode(m_scene.rootNode);
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
            ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&m_scene.lights[i].color));
            ImGui::DragFloat("Intensity", &m_scene.lights[i].intensity, 0.01f, 0.0f, FLT_MAX);

            ImGui::Separator();
            ImGui::PopID();
        }
        ImGui::End();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Viewport" , nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar);

        m_viewportWidth = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
        m_viewportHeight = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);

        if (const auto image = m_renderer.getImage())
            ImGui::Image(image->getDescriptorSet(), { static_cast<float>(image->getWidth()), static_cast<float>(image->getHeight()) },
                ImVec2(0, 1), ImVec2(1, 0));

        ImGui::End();
        ImGui::PopStyleVar();

        Render();
    }

	// Very lazy way to import and export scene data
    void ImportScene()
    {
        m_scene = Utils::importScene("scene.json");
        m_camera = m_scene.camera;
        m_renderer.setSettings(m_scene.settings);
        m_renderer.resetFrameIndex();
    }

	void ExportScene()
	{
		m_scene.camera = m_camera;
		m_scene.settings = m_renderer.getSettings();
		Utils::exportScene(m_scene, "scene.json");
	}

    void Render()
    {
        Timer timer;

        m_renderer.onResize(m_viewportWidth, m_viewportHeight);
        m_camera.Resize(m_viewportWidth, m_viewportHeight);
        m_renderer.Render(m_camera, m_scene);

        m_lastRenderTime = timer.ElapsedMS();
    }

	Renderer GetRenderer() const { return m_renderer; }

    Scene GetScene() const { return m_scene; }
	void SetScene(const Scene& scene) { m_scene = scene; }

private:
    Scene m_scene;
    Renderer m_renderer;
    Camera m_camera;

    uint32_t m_viewportWidth = 0, m_viewportHeight = 0;
    float m_lastRenderTime = 0.0f;

    void initializeScene()
    {
        std::shared_ptr<SceneNode> root = m_scene.rootNode;

        Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, 0);
        root->addSphere(sphere1);

        std::shared_ptr<SceneNode> childNode1 = std::make_shared<SceneNode>("ChildNode1");
        childNode1->setPosition(glm::vec3(2.0f, 0.0f, 0.0f));
        Sphere sphere2(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, 1);
        childNode1->addSphere(sphere2);
        root->addChild(childNode1);

        std::shared_ptr<SceneNode> grandChildNode = std::make_shared<SceneNode>("GrandChildNode");
        grandChildNode->setPosition(glm::vec3(0.0f, 2.0f, 0.0f));
        Sphere sphere3(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, 2);
        grandChildNode->addSphere(sphere3);
        childNode1->addChild(grandChildNode);

        Material mat1(glm::vec3(1.022f, 0.782f, 0.344f), 1.0f, 0.0f, glm::vec3(0.0f), 0.0f, 0);
        m_scene.materials.push_back(mat1);

        Material mat2(glm::vec3(1.0f, 0.0f, 0.0f), 0.3f, 0.0f, glm::vec3(0.0f), 0.0f, 1);
        m_scene.materials.push_back(mat2);

        Material mat3(glm::vec3(0.972f, 0.960f, 0.915f), 0.25f, 1.0f, glm::vec3(0.0f), 0.0f, 2);
        m_scene.materials.push_back(mat3);

        Light light1(glm::vec3(10.0f, 10.0f, 0.0f), glm::vec3(1.0f), 1.0f);
        m_scene.lights.push_back(light1);
    }
};

Application* createApplication(int argc, char** argv)
{
    Specs spec;
    spec.name = "Ataraxia";

    auto app = new Application(spec);
    app->pushLayer<Ataraxia>();
    app->setMenubarCallback([app]()
    {
        if (ImGui::BeginMenu("Add"))
        {
            if (ImGui::MenuItem("Sphere"))
            {
				const auto layer = app->getLayer<Ataraxia>();
				layer->GetScene().rootNode->addSphere(Sphere(glm::vec3(0.0f), 1.0f, 0));
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Fullscreen", "F11"))
                app->toggleFullscreen();

            if (ImGui::MenuItem("Export Scene"))
            {
                const auto layer = app->getLayer<Ataraxia>();
				layer->ExportScene();
            }

            if (ImGui::MenuItem("Import Scene"))
            {
				const auto layer = app->getLayer<Ataraxia>();
				layer->ImportScene();
            }

            if (ImGui::MenuItem("Exit"))
                app->close();

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help"))
        {
            if (ImGui::MenuItem("About"))
            {
                ImGui::OpenPopup("About");
            }
            if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::Text("Ataraxia Alpha");
                ImGui::Text("A simple Vulkan path-tracer");
                ImGui::Text("Created by nesk");
                ImGui::Separator();
                ImGui::Text("Press ESC to close");
                ImGui::EndPopup();
            }
            ImGui::EndMenu();
        }
    });
    return app;
}
