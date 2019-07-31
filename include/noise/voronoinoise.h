#pragma once
#ifndef VORONOI_NOISE_H
#define VORONOI_NOISE_H

#include "common.h"

#define DEFAULT_VORONOI_FREQUENCY 1.0
#define DEFAULT_VORONOI_DISPLACEMENT 1.0
#define DEFAULT_VORONOI_SEED 0
#define DEFAULT_VORONOI_ENABLE_DISTANCE 0

struct VoronoiNoise
{
    double frequency;
    double displacement;
    int seed;
    unsigned char enable_distance;
};

void voronoi_noise_init(struct VoronoiNoise *voronoi_noise)
{
    voronoi_noise->frequency = DEFAULT_VORONOI_FREQUENCY;
    voronoi_noise->displacement = DEFAULT_VORONOI_DISPLACEMENT;
    voronoi_noise->seed = DEFAULT_VORONOI_SEED;
    voronoi_noise->enable_distance = DEFAULT_VORONOI_ENABLE_DISTANCE;
}

static inline double voronoi_noise_eval_3d(struct VoronoiNoise *voronoi_noise, double x, double y, double z)
{
    x *= voronoi_noise->frequency;
    y *= voronoi_noise->frequency;
    z *= voronoi_noise->frequency;

    int xInt = (x > 0.0 ? (int)x : (int)x - 1);
    int yInt = (y > 0.0 ? (int)y : (int)y - 1);
    int zInt = (z > 0.0 ? (int)z : (int)z - 1);

    double minDist = 2147483647.0;
    double xCandidate = 0;
    double yCandidate = 0;
    double zCandidate = 0;

    for (int zCur = zInt - 2; zCur <= zInt + 2; zCur++)
    {
        for (int yCur = yInt - 2; yCur <= yInt + 2; yCur++)
        {
            for (int xCur = xInt - 2; xCur <= xInt + 2; xCur++)
            {
                double xPos = xCur + value_noise_3d(xCur, yCur, zCur, voronoi_noise->seed);
                double yPos = yCur + value_noise_3d(xCur, yCur, zCur, voronoi_noise->seed + 1);
                double zPos = zCur + value_noise_3d(xCur, yCur, zCur, voronoi_noise->seed + 2);
                double xDist = xPos - x;
                double yDist = yPos - y;
                double zDist = zPos - z;
                double dist = xDist * xDist + yDist * yDist + zDist * zDist;

                if (dist < minDist)
                {
                    minDist = dist;
                    xCandidate = xPos;
                    yCandidate = yPos;
                    zCandidate = zPos;
                }
            }
        }
    }

    double value;
    if (voronoi_noise->enable_distance)
    {
        // Determine the distance to the nearest seed point.
        double xDist = xCandidate - x;
        double yDist = yCandidate - y;
        double zDist = zCandidate - z;
        value = (sqrt(xDist * xDist + yDist * yDist + zDist * zDist)) * SQRT_3 - 1.0;
    }
    else
    {
        value = 0.0;
    }

    return value + (voronoi_noise->displacement * (double)value_noise_3d(fast_floor(xCandidate), fast_floor(yCandidate), fast_floor(zCandidate), voronoi_noise->seed));
}

#endif
