#pragma once

#include <cassert>

#include "types.h"

#include "probEstimator.h"

struct TTokenReader;
struct TFileWriter;

typedef std::vector<float> TFloatVector;

struct TNeuralEstimator : public IProbEstimator
{
    size_t m_inputSize;

    struct TNeuron
    {
        struct TSinapse
        {
            ui16 m_index;
            float m_weight;

            TSinapse()
            {
            }

            TSinapse(ui16 index, float weight)
                : m_index(index)
                , m_weight(weight)
            {
            }
        };
        typedef std::vector<TSinapse> TSinapses;
        TSinapses m_sinapses;

        struct TInvertedSinapse
        {
            ui16 m_neuronIndex;
            ui16 m_sinapseIndex;

            TInvertedSinapse()
            {
            }

            TInvertedSinapse(ui16 neuronIndex, ui16 sinapseIndex)
                : m_neuronIndex(neuronIndex)
                , m_sinapseIndex(sinapseIndex)
            {
            }
        };
        typedef std::vector<TInvertedSinapse> TInvertedSinapses;
        TInvertedSinapses m_invertedSinapses;

        void AddSinapse(ui16 inputIndex, float weight = 0.f);

        void Save(TFileWriter& fOut) const;
        void Load(TTokenReader& fIn);
    };
    typedef std::vector<TNeuron> TNeurons;
    TNeurons m_neurons;

    void SetInputSize(size_t inputSize)
    {
        m_inputSize = inputSize;
        assert(m_neurons.empty());
        for (size_t i = 0; i < inputSize; ++i)
        {
            m_neurons.push_back(TNeuron());
        }
    }

    void Add(const TNeuron& neuron)
    {
        m_neurons.push_back(neuron);
    }

    void Prepare()
    {
        for (size_t i = 0; i < m_neurons.size(); ++i)
        {
            const TNeuron& neuron = m_neurons[i];
            for (size_t j = 0; j < neuron.m_sinapses.size(); ++j)
            {
                const TNeuron::TSinapse& sinapse = neuron.m_sinapses[j];
                assert(sinapse.m_index < i);
                m_neurons[sinapse.m_index].m_invertedSinapses.push_back( TNeuron::TInvertedSinapse(i, j) );
            }
        }
    }

    void CalculateValues(const TFloatVector& input, TFloatVector* result) const;

    float GetOutput(const TFloatVector& input) const
    {
        TFloatVector data;
        CalculateValues(input, &data);
        return data.back();
    }

    float Estimate(const TPicture& input) const;

    void Inflate();

    void BackPropagation(const TFloatVector& input, float targetOutput, size_t iteration)
    {
        TFloatVector data;
        CalculateValues(input, &data);
        TFloatVector delta(Size());
        delta.back() = data.back()*(1.f - data.back())*(targetOutput - data.back());
        for (size_t i = m_neurons.size() - 2; i >= m_inputSize; --i)
        {
            float sum = 0.f;
            for (size_t j = 0; j < m_neurons[i].m_invertedSinapses.size(); ++j)
            {
                const TNeuron::TInvertedSinapse& is = m_neurons[i].m_invertedSinapses[j];
                sum += delta[is.m_neuronIndex]*m_neurons[is.m_neuronIndex].m_sinapses[is.m_sinapseIndex].m_weight;
            }
            delta[i] = data[i]*(1.f - data[i])*sum;
        }
        const float learnRate = 0.5f/sqrtf(static_cast<float>(iteration) + 1.f);
        for (size_t i = m_inputSize; i < m_neurons.size(); ++i)
        {
            // printf("%d %f %f %d %d\n", (int)i, data[i], delta[i], m_neurons[i].m_sinapses.size(), m_neurons[i].m_invertedSinapses.size());
            TNeuron& neuron = m_neurons[i];
            for (size_t j = 0; j < neuron.m_sinapses.size(); ++j)
            {
                TNeuron::TSinapse& s = neuron.m_sinapses[j];
                // printf("\t%f %f %f %f %f\n", s.m_weight, learnRate*delta[i]*data[s.m_index], learnRate, delta[i], data[s.m_index]);
                s.m_weight += learnRate*delta[i]*data[s.m_index];
            }
        }
    }

    size_t Size() const
    {
        return m_neurons.size();
    }

    void Save(const std::string& filename) const;
	void Load(const std::string& filename);
};