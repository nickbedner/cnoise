#pragma once
#ifndef WORLEY_NOISE_H
#define WORLEY_NOISE_H

/*#include "common.h"

#define DENSITY_ADJUSTMENT 0.398150

struct WorleyNoise
{
    unsigned char perm[256];
};

void worley_noise_init(struct WorleyNoise *worley_noise, long seed)
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
        worley_noise->perm[i] = source[r];
        source[r] = source[i];
    }
}

static inline void worley_add_samples(struct WorleyNoise *worley_noise, int32_t xi, int32_t yi, int32_t zi, size_t max_order, double at[3], double *F, double (*delta)[3], uint32_t *ID)
{
    double dx, dy, dz, fx, fy, fz, d2;
    int32_t count, i, j, index;
    uint32_t seed, this_id;
    seed = 702395077 * xi + 915488749 * yi + 2120969693 * zi;
    count = worley_noise->perm[seed >> 24];
    seed = 1402024253 * seed + 586950981;
    for (j = 0; j < count; j++)
    {
        this_id = seed;
        seed = 1402024253 * seed + 586950981;

        fx = (seed + 0.5) * (1.0 / 4294967296.0);
        seed = 1402024253 * seed + 586950981;
        fy = (seed + 0.5) * (1.0 / 4294967296.0);
        seed = 1402024253 * seed + 586950981;
        fz = (seed + 0.5) * (1.0 / 4294967296.0);
        seed = 1402024253 * seed + 586950981;

        dx = xi + fx - at[0];
        dy = yi + fy - at[1];
        dz = zi + fz - at[2];

        d2 = dx * dx + dy * dy + dz * dz;

        if (d2 < F[max_order - 1])
        {
            index = max_order;
            while (index > 0 && d2 < F[index - 1])
                index--;

            for (i = max_order - 2; i >= index; i--)
            {
                F[i + 1] = F[i];
                ID[i + 1] = ID[i];
                delta[i + 1][0] = delta[i][0];
                delta[i + 1][1] = delta[i][1];
                delta[i + 1][2] = delta[i][2];
            }

            F[index] = d2;
            ID[index] = this_id;
            delta[index][0] = dx;
            delta[index][1] = dy;
            delta[index][2] = dz;
        }
    }

    return;
}

void worley_noise_eval3d(struct WorleyNoise *worley_noise, double at[3], size_t max_order, double *F, double (*delta)[3], uint32_t *ID)
{
    double x2, y2, z2, mx2, my2, mz2;
    double new_at[3];
    int32_t int_at[3], i;

    for (i = 0; i < max_order; i++)
        F[i] = 999999.9;

    new_at[0] = DENSITY_ADJUSTMENT * at[0];
    new_at[1] = DENSITY_ADJUSTMENT * at[1];
    new_at[2] = DENSITY_ADJUSTMENT * at[2];

    int_at[0] = fast_floor(new_at[0]);
    int_at[1] = fast_floor(new_at[1]);
    int_at[2] = fast_floor(new_at[2]);

    worley_add_samples(worley_noise, int_at[0], int_at[1], int_at[2], max_order, new_at, F, delta, ID);

    x2 = new_at[0] - int_at[0];
    y2 = new_at[1] - int_at[1];
    z2 = new_at[2] - int_at[2];
    mx2 = (1.0 - x2) * (1.0 - x2);
    my2 = (1.0 - y2) * (1.0 - y2);
    mz2 = (1.0 - z2) * (1.0 - z2);
    x2 *= x2;
    y2 *= y2;
    z2 *= z2;

    if (x2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1], int_at[2], max_order, new_at, F, delta, ID);
    if (y2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1] - 1, int_at[2], max_order, new_at, F, delta, ID);
    if (z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1], int_at[2] - 1, max_order, new_at, F, delta, ID);

    if (mx2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1], int_at[2], max_order, new_at, F, delta, ID);
    if (my2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1] + 1, int_at[2], max_order, new_at, F, delta, ID);
    if (mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1], int_at[2] + 1, max_order, new_at, F, delta, ID);

    if (x2 + y2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1] - 1, int_at[2], max_order, new_at, F, delta, ID);
    if (x2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1], int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (y2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1] - 1, int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (mx2 + my2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1] + 1, int_at[2], max_order, new_at, F, delta, ID);
    if (mx2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1], int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (my2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1] + 1, int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (x2 + my2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1] + 1, int_at[2], max_order, new_at, F, delta, ID);
    if (x2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1], int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (y2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1] - 1, int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (mx2 + y2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1] - 1, int_at[2], max_order, new_at, F, delta, ID);
    if (mx2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1], int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (my2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0], int_at[1] + 1, int_at[2] - 1, max_order, new_at, F, delta, ID);

    if (x2 + y2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1] - 1, int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (x2 + y2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1] - 1, int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (x2 + my2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1] + 1, int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (x2 + my2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] - 1, int_at[1] + 1, int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (mx2 + y2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1] - 1, int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (mx2 + y2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1] - 1, int_at[2] + 1, max_order, new_at, F, delta, ID);
    if (mx2 + my2 + z2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1] + 1, int_at[2] - 1, max_order, new_at, F, delta, ID);
    if (mx2 + my2 + mz2 < F[max_order - 1])
        worley_add_samples(worley_noise, int_at[0] + 1, int_at[1] + 1, int_at[2] + 1, max_order, new_at, F, delta, ID);

    for (i = 0; i < max_order; i++)
    {
        F[i] = sqrt(F[i]) * (1.0 / DENSITY_ADJUSTMENT);
        delta[i][0] *= (1.0 / DENSITY_ADJUSTMENT);
        delta[i][1] *= (1.0 / DENSITY_ADJUSTMENT);
        delta[i][2] *= (1.0 / DENSITY_ADJUSTMENT);
    }

    return;
}*/

#endif
