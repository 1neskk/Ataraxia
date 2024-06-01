#include "Input.h"

#include "../Application.h"

#include <GLFW/glfw3.h>

namespace Input
{
    bool Input::IsKeyPressed(KeyCode keycode)
    {
        GLFWwindow* window = Application::get().getWindow();
        auto state = glfwGetMouseButton(window, static_cast<int>(keycode));
        return state == GLFW_PRESS || state == GLFW_REPEAT;
    }

    bool Input::IsMouseButtonPressed(MouseButton button)
    {
        GLFWwindow* window = Application::get().getWindow();
        auto state = glfwGetMouseButton(window, static_cast<int>(button));
        return state == GLFW_PRESS;
    }

    glm::vec2 Input::GetMousePosition()
    {
        GLFWwindow* window = Application::get().getWindow();
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        return { static_cast<float>(xpos), static_cast<float>(ypos) };
    }

    void Input::SetCursorState(CursorState state)
    {
        GLFWwindow* window = Application::get().getWindow();
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL + static_cast<int>(state));
    }
}
