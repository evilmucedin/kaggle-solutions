#include "fileIO.h"
#include "random.h"
#include "picture.h"
#include "math.h"

#include "neural.h"

void TNeuralEstimator::TNeuron::AddSinapse(ui16 inputIndex, float weight)
{
	weight *= 10.f;
	if (0.f == weight)
	{
		weight = (Rand01() - 0.5f)*2.f;
	}
	m_sinapses.push_back( TSinapse(inputIndex, weight) );
}


void TNeuralEstimator::TNeuron::Load(TTokenReader& fIn)
{
	const ui32 len = fIn.NextTokenUi32();
	if (len > m_sinapses.size())
	{
		throw TException("bad number of sinapses");
	}

	for (ui32 i = 0; i < len; ++i)
	{
		m_sinapses[i].m_weight = fIn.NextTokenFloat();
	}
}

void TNeuralEstimator::TNeuron::Save(TFileWriter& fOut) const
{
	fOut.Write(std::string("\t") + std::to_string(m_sinapses.size()) + "\n");
	for (size_t i = 0; i < m_sinapses.size(); ++i)
	{
		fOut.Write(std::string("\t\t") + std::to_string(m_sinapses[i].m_weight) + "\n");
	}
}

void TNeuralEstimator::CalculateValues(const TFloatVector& input, TFloatVector* result) const
{
	assert(input.size() == m_inputSize);
	result->resize(Size());
	for (size_t i = 0; i < input.size(); ++i)
	{
		(*result)[i] = input[i];
	}
	for (size_t i = m_inputSize; i < m_neurons.size(); ++i)
	{
		float value = 0.f;
		const TNeuron& neuron = m_neurons[i];
		for (size_t j = 0; j < neuron.m_sinapses.size(); ++j)
		{
			const TNeuron::TSinapse& sin = neuron.m_sinapses[j];
			value += sin.m_weight*(*result)[sin.m_index];
		}
		(*result)[i] = Sigmoid(value);
	}

}

float TNeuralEstimator::Estimate(const TPicture& input) const
{
	return GetOutput(input.AsVector());
}

void TNeuralEstimator::Inflate()
{
	for (size_t i = m_inputSize; i < m_neurons.size(); ++i)
	{
		TNeuron& neuron = m_neurons[i];
		for (size_t j = 0; j < neuron.m_sinapses.size(); ++j)
		{
			neuron.m_sinapses[j].m_weight *= 1.f + (Rand01() - 0.6f)*0.00001f;
			neuron.m_sinapses[j].m_weight += (Rand01() - 0.6f)*0.0000001f;
		}
	}
}

void TNeuralEstimator::Load(const std::string& filename)
{
	TTokenReader fIn(filename);
	const ui32 len = fIn.NextTokenUi32();
	if (len > m_neurons.size())
	{
		throw TException("bad neuro net input");
	}
	for (ui32 i = 0; i < len; ++i)
	{
		m_neurons[i].Load(fIn);
	}
}

void TNeuralEstimator::Save(const std::string& filename) const
{
	TFileWriter fOut(filename);
	fOut.Write( std::to_string(m_neurons.size()) + "\n" );
	for (size_t i = 0; i < m_neurons.size(); ++i)
	{
		m_neurons[i].Save(fOut);
	}
}

