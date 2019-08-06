#pragma once
#ifndef BILLOW_NOISE_H
#define BILLOW_NOISE_H

#include "common.h"

#define DEFAULT_BILLOW_FREQUENCY 1.0
#define DEFAULT_BILLOW_LACUNARITY 2.0
#define DEFAULT_BILLOW_PERSISTENCE 0.5
#define DEFAULT_BILLOW_OCTAVE_COUNT 6
#define DEFAULT_BILLOW_SEED 0
#define DEFAULT_BILLOW_QUALITY QUALITY_STANDARD

struct BillowNoise
{
    double frequency;
    double lacunarity;
    double persistence;
    unsigned char octave_count;
    int seed;
    enum NoiseQuality noise_quality;
};

static inline void billow_noise_init(struct BillowNoise *billow_noise)
{
    billow_noise->frequency = DEFAULT_BILLOW_FREQUENCY;
    billow_noise->lacunarity = DEFAULT_BILLOW_LACUNARITY;
    billow_noise->persistence = DEFAULT_BILLOW_PERSISTENCE;
    billow_noise->octave_count = DEFAULT_BILLOW_OCTAVE_COUNT;
    billow_noise->seed = DEFAULT_BILLOW_SEED;
    billow_noise->noise_quality = DEFAULT_BILLOW_QUALITY;
}

static inline double billow_noise_eval_3d(struct BillowNoise *billow_noise, double x, double y, double z)
{
    double value = 0.0;
    double signal = 0.0;
    double curPersistence = 1.0;
    double nx, ny, nz;
    int curSeed;

    x *= billow_noise->frequency;
    y *= billow_noise->frequency;
    z *= billow_noise->frequency;

    for (int curOctave = 0; curOctave < billow_noise->octave_count; curOctave++)
    {
        nx = make_int_32_range(x);
        ny = make_int_32_range(y);
        nz = make_int_32_range(z);

        curSeed = (billow_noise->seed + curOctave) & 0xffffffff;
        signal = gradient_coherent_noise_3d(nx, ny, nz, curSeed, billow_noise->noise_quality);
        signal = 2.0 * fabs(signal) - 1.0;
        value += signal * curPersistence;

        x *= billow_noise->lacunarity;
        y *= billow_noise->lacunarity;
        z *= billow_noise->lacunarity;
        curPersistence *= billow_noise->persistence;
    }

    value += 0.5;

    return value;
}

#endif
