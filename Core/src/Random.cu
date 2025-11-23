#include <device_launch_parameters.h>

#include "Random.h"

namespace Random {
__global__ void initRandomStates(curandState* state, unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    Random::Init(state, seed, id);
}
}  // namespace Random
