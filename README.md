# PathTracingEngine

![image](https://github.com/1neskk/PathTracingEngine/assets/113075816/42991693-2374-4779-81e7-b2159f28cf6b)

This project uses thirdparty libraries such as [GLFW](https://github.com/glfw/glfw), [GLM](https://github.com/g-truc/glm), [Dear ImGui](https://github.com/ocornut/imgui) and [stb](https://github.com/nothings/stb)

**Officially supports Windows 10/11 with Visual Studio 2022 and Linux (tested on the LTS kernel with Wayland)**

## Requirements
- [Vulkan SDK](https://vulkan.lunarg.com/)
- NVIDIA GPU
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Get Started

### Linux

1. Clone recursively: `git clone --recursive https://github.com/1neskk/PathTracingEngine`

2. Build the project: `make build`

3. Run the resulting executable: `./build/PathTracingEngine`

### Windows

1. Clone recursively: `git clone --recursive https://github.com/1neskk/PathTracingEngine`

2. Build the project: `mkdir build && cmake -B build -A x64 `

3. Open the resulting solution file: `cd build && start PathTracingEngine.sln`

#### To Do
- [x] Material system
- [ ] Environment mapping
- [ ] HDR lighting
