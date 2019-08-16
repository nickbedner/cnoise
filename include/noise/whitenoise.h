#pragma once
#ifndef WHITE_NOISE_H
#define WHITE_NOISE_H

#include "common.h"

#define DEFAULT_WHITE_SEED 0

struct WhiteNoise
{
    int seed;
};

static inline void white_noise_init(struct WhiteNoise *white_noise)
{
    white_noise->seed = DEFAULT_WHITE_SEED;
}

// TODO: Check for artifacts and benchmark
static inline float white_noise_eval_3d(struct WhiteNoise *white_noise, float x, float y, float z)
{
    return (float)((X_NOISE_GEN * *(int *)&x + Y_NOISE_GEN * *(int *)&y + Z_NOISE_GEN * *(int *)&z + SEED_NOISE_GEN * white_noise->seed) & 0xffffffff * 16807) / 2147483647.5 - 1.0;
}

#endif
