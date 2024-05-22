#pragma once

#include "Vec3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../thirdparty/stb/stb_image_write.h"

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

struct Sphere
{
    Vec3 center;
    float radius;
};

