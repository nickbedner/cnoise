#pragma once
#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#include "common.h"
#include <immintrin.h>

#define DEFAULT_PERLIN_FREQUENCY 1.0
#define DEFAULT_PERLIN_LACUNARITY 2.0
#define DEFAULT_PERLIN_PERSISTENCE 0.5
#define DEFAULT_PERLIN_OCTAVE_COUNT 1
#define DEFAULT_PERLIN_SEED 0
#define DEFAULT_PERLIN_QUALITY QUALITY_STANDARD

struct PerlinNoise
{
    float frequency;
    float lacunarity;
    float persistence;
    unsigned char octave_count;
    int seed;
    enum NoiseQuality noise_quality;
};

static inline void perlin_noise_init(struct PerlinNoise *perlin_noise)
{
    perlin_noise->frequency = DEFAULT_PERLIN_FREQUENCY;
    perlin_noise->lacunarity = DEFAULT_PERLIN_LACUNARITY;
    perlin_noise->persistence = DEFAULT_PERLIN_PERSISTENCE;
    perlin_noise->octave_count = DEFAULT_PERLIN_OCTAVE_COUNT;
    perlin_noise->seed = DEFAULT_PERLIN_SEED;
    perlin_noise->noise_quality = DEFAULT_PERLIN_QUALITY;
}

static inline float perlin_noise_eval_3d(struct PerlinNoise *perlin_noise, float x, float y, float z)
{
    float value = 0.0;
    float signal = 0.0;
    float cur_persistence = 1.0;
    float nx, ny, nz;
    int cur_seed;

    x *= perlin_noise->frequency;
    y *= perlin_noise->frequency;
    z *= perlin_noise->frequency;

    for (int cur_octave = 0; cur_octave < perlin_noise->octave_count; cur_octave++)
    {
        nx = make_int_32_range(x);
        ny = make_int_32_range(y);
        nz = make_int_32_range(z);

        cur_seed = (perlin_noise->seed + cur_octave) & 0xffffffff;
        signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, perlin_noise->noise_quality);
        value += signal * cur_persistence;

        x *= perlin_noise->lacunarity;
        y *= perlin_noise->lacunarity;
        z *= perlin_noise->lacunarity;
        cur_persistence *= perlin_noise->persistence;
    }

    return value;
}

// Note: Dimension must be multiple of 8 for SIMD to work properly
static inline void perlin_noise_eval_3d_vec_256(struct PerlinNoise *perlin_noise, float x, float y, float z, float *values, size_t x_size, size_t y_size, size_t z_size)
{
    for (int x_dim = 0; x_dim < x_size; x_dim += 8)
    {
        for (int y_dim = 0; y_dim < y_size; y_dim++)
        {
            for (int z_dim = 0; z_dim < z_size; z_dim++)
            {
                __m256 value = _mm256_set1_ps(0.0);
                __m256 signal = _mm256_set1_ps(0.0);
                __m256 cur_persistence = _mm256_set1_ps(1.0);
                //__m256 nx, ny, nz;
                int cur_seed;

                // Might need to use spacing variable instead of hard coding values
                __m256 x_vec = _mm256_set_ps((x + x_dim) / x_size, (x + x_dim + 1.0) / x_size, (x + x_dim + 2.0) / x_size, (x + x_dim + 3.0) / x_size, (x + x_dim + 4.0) / x_size, (x + x_dim + 5.0) / (x_size, x + x_dim + 6.0) / (x_size, x + x_dim + 7.0) / x_size);
                __m256 y_vec = _mm256_set1_ps((y + y_dim) / y_size);
                __m256 z_vec = _mm256_set1_ps((z + z_dim) / z_size);

                const __m256 frequency_scalar = _mm256_set1_ps(perlin_noise->frequency);
                x_vec = _mm256_mul_ps(x_vec, frequency_scalar);
                y_vec = _mm256_mul_ps(y_vec, frequency_scalar);
                z_vec = _mm256_mul_ps(z_vec, frequency_scalar);

                for (int cur_octave = 0; cur_octave < perlin_noise->octave_count; cur_octave++)
                {
                    // Note: only need to check last value x_vec[0] to see if in range(although probably better to remove range requirement)
                    //nx = make_int_32_range(x);
                    //ny = make_int_32_range(y);
                    //nz = make_int_32_range(z);

                    cur_seed = (perlin_noise->seed + cur_octave) & 0xffffffff;
                    signal = gradient_coherent_noise_3d_vec_256(nx, ny, nz, cur_seed, perlin_noise->noise_quality);
                    value = _mm256_add_ps(value, _mm256_mul_ps(signal, cur_persistence));

                    x *= perlin_noise->lacunarity;
                    y *= perlin_noise->lacunarity;
                    z *= perlin_noise->lacunarity;

                    const __m256 persistence_scalar = _mm256_set1_ps(perlin_noise->persistence);
                    cur_persistence = _mm256_mul_ps(cur_persistence, persistence_scalar);
                }

                                //return value;
            }
        }
    }
}

#endif
