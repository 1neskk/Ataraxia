#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include "Scene.h"

using json = nlohmann::json;

namespace Utils
{
    // Export
    json serializeScene(const Scene& scene);
    void exportScene(const Scene& scene, const std::string& filename);

    // Import
    Scene deserializeScene(const json& j);
    Scene importScene(const std::string& filename);
}
