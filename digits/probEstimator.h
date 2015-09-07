#pragma once

#include <vector>

struct TPicture;

struct IProbEstimator
{
    virtual float Estimate(const TPicture& picture) const = 0;

    virtual ~IProbEstimator()
    {
    }
};

typedef std::vector<IProbEstimator*> IProbEstimators;
