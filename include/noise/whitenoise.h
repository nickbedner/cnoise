#pragma once
#ifndef WHITE_NOISE_H
#define WHITE_NOISE_H

#include "common.h"

#define DEFAULT_WHITE_SEED 0

#define A_VAL 16807
#define MOD_VAL 2147483647

struct WhiteNoise
{
    int seed;
};

void white_noise_init(struct WhiteNoise *white_noise)
{
    white_noise->seed = DEFAULT_WHITE_SEED;
}

double white_noise_eval_3d(struct WhiteNoise *white_noise, double x, double y, double z)
{
    return fmod(white_noise->seed * (x + white_noise->seed) * (y + white_noise->seed) * (z + white_noise->seed) * A_VAL, MOD_VAL);
}

#endif
