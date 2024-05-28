#include "Application.h"
#include "main.h"
#include "Image.h"

class CUDARayTracer final : public Layer
{
public:
    virtual void onImGuiRender() override
    {
        ImGui::CreateContext();
        const auto& io = ImGui::GetIO();

        Style::theme();
        ImGui::Begin("Example");
        if (ImGui::Button("Hello, World!"))
            std::cout << "Hello, World!" << std::endl;
        ImGui::End();
    }
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
