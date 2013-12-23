#pragma once

template<typename T>
inline T Sqr(T x)
{
    return x*x;
}

inline float Sigmoid(float value)
{
    return 1.f / (1.f + expf(-value));
}
