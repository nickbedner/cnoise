#pragma once
#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#include "common.h"

#define DEFAULT_PERLIN_FREQUENCY 1.0
#define DEFAULT_PERLIN_LACUNARITY 2.0
#define DEFAULT_PERLIN_PERSISTENCE 0.5
#define DEFAULT_PERLIN_OCTAVE_COUNT 1
#define DEFAULT_PERLIN_SEED 0
#define DEFAULT_PERLIN_QUALITY QUALITY_STANDARD

struct PerlinNoise
{
    double frequency;
    double lacunarity;
    double persistence;
    unsigned char octave_count;
    int seed;
    enum NoiseQuality noise_quality;
};

void perlin_noise_init(struct PerlinNoise *perlin_noise)
{
    perlin_noise->frequency = DEFAULT_PERLIN_FREQUENCY;
    perlin_noise->lacunarity = DEFAULT_PERLIN_LACUNARITY;
    perlin_noise->persistence = DEFAULT_PERLIN_PERSISTENCE;
    perlin_noise->octave_count = DEFAULT_PERLIN_OCTAVE_COUNT;
    perlin_noise->seed = DEFAULT_PERLIN_SEED;
    perlin_noise->noise_quality = DEFAULT_PERLIN_QUALITY;
}

double perlin_noise_eval_3d(struct PerlinNoise *perlin_noise, double x, double y, double z)
{
    double value = 0.0;
    double signal = 0.0;
    double curPersistence = 1.0;
    double nx, ny, nz;
    int curSeed;

    x *= perlin_noise->frequency;
    y *= perlin_noise->frequency;
    z *= perlin_noise->frequency;

    for (int curOctave = 0; curOctave < perlin_noise->octave_count; curOctave++)
    {
        nx = make_int_32_range(x);
        ny = make_int_32_range(y);
        nz = make_int_32_range(z);

        curSeed = (perlin_noise->seed + curOctave) & 0xffffffff;
        signal = gradient_coherent_noise_3d(nx, ny, nz, curSeed, perlin_noise->noise_quality);
        value += signal * curPersistence;

        x *= perlin_noise->lacunarity;
        y *= perlin_noise->lacunarity;
        z *= perlin_noise->lacunarity;
        curPersistence *= perlin_noise->persistence;
    }

    return value;
}

#endif
