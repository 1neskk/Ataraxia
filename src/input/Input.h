#pragma once

#include "KeyCodes.h"
#include "vec/Vec2.h"

namespace Input
{
    class Input
    {
    public:
        static bool IsKeyPressed(KeyCode key);
        static bool IsMouseButtonPressed(MouseButton button);
        static Vec2 GetMousePosition();
        static void SetCursorState(CursorState state);
    };
}

