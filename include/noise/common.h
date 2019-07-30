#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static inline double fade(double t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

static inline double lerp(double t, double a, double b)
{
    return a + t * (b - a);
}

static inline double grad(int hash, double x, double y, double z)
{
    int h = hash & 15;
    double u = h < 8 ? x : y, v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

static inline int fast_floor(double x)
{
    int xi = (int)x;
    return x < xi ? xi - 1 : xi;
}

#endif
