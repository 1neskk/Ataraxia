#include "Application.h"
#include "main.h"
#include "Image.h"

class ExampleApp final : public Layer
{
public:
    virtual void onImGuiRender() override
    {
        ImGui::Begin("Example");
        ImGui::Button("Hello, World!");
        ImGui::End();

        //ImGui::ShowDemoWindow();
    }
};

Application* createApplication(int argc, char** argv)
{
    Specs spec;
    spec.name = "App";

    auto app = new Application(spec);
    app->pushLayer<ExampleApp>();
    // app->setMenubarCallback([app]()
    // {
    //     if (ImGui::BeginMenu("File"))
    //     {
    //         if (ImGui::MenuItem("Exit"))
    //             app->close();
    //         ImGui::EndMenu();
    //     }
    // });
    return app;
}
