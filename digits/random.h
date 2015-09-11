#pragma once

#include <vector>

inline float Rand01()
{
    return ((float)rand())/RAND_MAX;
}

template<typename T>
void Shuffle(std::vector<T>& v)
{
    for (ssize_t i = static_cast<ssize_t>(v.size()) - 1; i >= 1; --i)
    {
        size_t index = rand() % i;
        swap(v[index], v[i]);
    }
}
