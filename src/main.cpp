#include "Application.h"
#include "main.h"
#include "Image.h"

class ExampleApp final : public Layer
{
public:
    ExampleApp()
    {

    }

    virtual void onUpdate() override
    {

    }

    virtual void onImGuiRender() override
    {
        ImGui::CreateContext();
        const ImGuiIO& io = ImGui::GetIO();
        Style::theme();

        ImGui::Begin("test");
        ImGui::Text("Hello, world!");
        ImGui::End();
    }

private:
};

Application* createApplication(int argc, char** argv)
{
    Specs spec;
    spec.name = "CUDA";

    auto app = new Application(spec);
    app->pushLayer<ExampleApp>();
    return app;
}
