#include "Random.h"

#include <device_launch_parameters.h>

namespace Random
{
    __global__ void initRandomStates(curandState* state, unsigned long long seed)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        Random::Init(state, seed, id);
    }
}
