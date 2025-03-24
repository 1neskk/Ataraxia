# Ataraxia Engine
This project is a GPU-accelerated path tracing engine built with CUDA and Vulkan. It is still in the early stages of development, in addition to the core rendering system, it also includes a scene system, a camera system, and a UI system. It is built with the intention of being a learning experience for me, and to serve as a foundation for future projects.

## Table of Contents
- [Features](#features)
- [Showcase](#showcase)
- [Building](#building)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## Features

### Core Rendering
- Real-time path tracing using CUDA
- Physically based rendering (PBR) with Cook-Torrance BRDF
- Support for diffuse and metallic materials with configurable properties:
  - Albedo
  - Roughness 
  - Metallic factor
  - Fresnel reflection (F0)
- Multiple importance sampling for efficient light sampling
- Cosine weighted hemisphere sampling
- GGX microfacet distribution
- Accumulation buffer for progressive refinement

### Scene System
- Sphere primitive support with configurable:
  - Position
  - Radius
  - Material assignment
- Point light system with:
  - Position
  - Color
  - Intensity
- Interactive camera with:
  - WASD movement
  - Mouse look
  - FOV control
  - Near/far clip planes

### Technical Details
- Hybrid rendering pipeline using Vulkan for display and CUDA for compute
- Multi-threaded ray direction computation
- ImGui-based UI for real-time parameter tuning
- GLFW windowing system
- GLM math library integration

### Performance
- Parallel ray tracing on GPU
- Configurable bounce depth
- Frame accumulation for noise reduction
- Real-time FPS counter and render time display

## Showcase

![Screenshot 2024-12-17 025805](https://github.com/user-attachments/assets/7485fcab-acbe-4df4-838b-8e5dbca2510b)

![Screenshot 2024-12-16 012411](https://github.com/user-attachments/assets/faf609f1-248b-4d8e-86bd-60c1fb726a5c)

---

## Building

**Currently supports only Windows 10/11 with Visual Studio 2022, to be tested on Linux.**

Requires:
- CUDA Toolkit (NVIDIA GPU required)
- Vulkan SDK
- GLFW, GLM, ImGui, stb and nlohmann's json (included as submodules)

#### Windows

1. Clone recursively: `git clone --recursive https://github.com/1neskk/Ataraxia`

2. Run the dependency installer if needed: `Scripts/dependencies.bat`

3. Run the build script: `Scripts/build.bat`

4. Open the resulting solution file: `cd build && start Ataraxia.sln`

## Usage

The engine provides an interactive viewport where you can:
- Move the camera using WASD keys and right mouse button
- Adjust material properties in real-time
- Configure light parameters
- Toggle accumulation for higher quality renders
- Reset frame accumulation when needed
- Monitor performance metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or improvements.
