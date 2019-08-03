#pragma once
#ifndef WHITE_NOISE_H
#define WHITE_NOISE_H

#include "common.h"

#define DEFAULT_WHITE_SEED 0

struct WhiteNoise
{
    int seed;
};

union MemorySwitch {
    unsigned long long_rep;
    double double_rep;
};

void white_noise_init(struct WhiteNoise *white_noise)
{
    white_noise->seed = DEFAULT_WHITE_SEED;
}

// TODO: Check for artifacts
double white_noise_eval_3d(struct WhiteNoise *white_noise, double x, double y, double z)
{
    union MemorySwitch x0, y0, z0;
    x0.double_rep = x;
    y0.double_rep = y;
    z0.double_rep = z;

    unsigned int vector_index = (X_NOISE_GEN * x0.long_rep + Y_NOISE_GEN * y0.long_rep + Z_NOISE_GEN * z0.long_rep + SEED_NOISE_GEN * white_noise->seed) & 0xffffffff;

    vector_index ^= (vector_index >> SHIFT_NOISE_GEN);
    vector_index &= 0xff;

    return random_vectors[vector_index][vector_index % 3];
}

#endif
