#pragma once
#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#include "common.h"

struct PerlinNoise
{
    unsigned char perm[256];
};

void perlin_noise_init(struct PerlinNoise *perlin_noise, long seed)
{
    short source[256];
    for (short i = 0; i < 256; i++)
        source[i] = i;

    seed = seed * 6364136223846793005l + 1442695040888963407l;
    seed = seed * 6364136223846793005l + 1442695040888963407l;
    seed = seed * 6364136223846793005l + 1442695040888963407l;

    for (int i = 255; i >= 0; i--)
    {
        seed = seed * 6364136223846793005l + 1442695040888963407l;
        int r = (int)((seed + 31) % (i + 1));
        if (r < 0)
            r += (i + 1);
        perlin_noise->perm[i] = source[r];
        source[r] = source[i];
    }
}

double perlin_noise_eval3d(struct PerlinNoise *perlin_noise, double x, double y, double z)
{
    int X = fast_floor(x) & 255,
        Y = fast_floor(y) & 255,
        Z = fast_floor(z) & 255;
    x -= fast_floor(x);
    y -= fast_floor(y);
    z -= fast_floor(z);
    double u = fade(x),
           v = fade(y),
           w = fade(z);
    int A = perlin_noise->perm[X] + Y, AA = perlin_noise->perm[A] + Z, AB = perlin_noise->perm[A + 1] + Z,
        B = perlin_noise->perm[X + 1] + Y, BA = perlin_noise->perm[B] + Z, BB = perlin_noise->perm[B + 1] + Z;

    return lerp(w, lerp(v, lerp(u, grad(perlin_noise->perm[AA], x, y, z), grad(perlin_noise->perm[BA], x - 1, y, z)), lerp(u, grad(perlin_noise->perm[AB], x, y - 1, z), grad(perlin_noise->perm[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, grad(perlin_noise->perm[AA + 1], x, y, z - 1), grad(perlin_noise->perm[BA + 1], x - 1, y, z - 1)),
                     lerp(u, grad(perlin_noise->perm[AB + 1], x, y - 1, z - 1),
                          grad(perlin_noise->perm[BB + 1], x - 1, y - 1, z - 1))));
}

#endif
