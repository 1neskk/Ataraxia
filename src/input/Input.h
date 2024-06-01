#pragma once

#include "KeyCodes.h"
#include <glm/glm.hpp>

namespace Input
{
    class Input
    {
    public:
        static bool IsKeyPressed(KeyCode key);
        static bool IsMouseButtonPressed(MouseButton button);
        static glm::vec2 GetMousePosition();
        static void SetCursorState(CursorState state);
    };
}

