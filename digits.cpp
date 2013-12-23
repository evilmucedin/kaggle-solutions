#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <vector>
#include <type_traits>

#include "easybmp/EasyBMP.h"

#include "gtest/gtest.h"

#include "fileIO.h"
#include "cmdline.h"
#include "csv.h"
#include "exceptions.h"
#include "str.h"
#include "random.h"

using namespace std;
using namespace testing;

struct TTokenReader
{
    TFileReader m_reader;

    TTokenReader(const std::string& filename)
        : m_reader(filename)
    {
    }

    static bool IsDelim(char ch)
    {
        return ('\t' == ch) || ('\r' == ch) || ('\n' == ch);
    }

    template<typename T>
    T NextToken()
    {
        T::Unimplemented();
    }

    std::string NextToken()
    {
        char now = m_reader.ReadChar();
        while (!m_reader.Eof() && IsDelim(now))
        {
            now = m_reader.ReadChar();
        }
        string result;
        if (!IsDelim(now))
        {
            result += now;
            while (!m_reader.Eof())
            {
                now = m_reader.ReadChar();
                if (!IsDelim(now))
                {
                    result += now;
                }
                else
                {
                    break;
                }
            }
        }
        return result;
    }

    ui32 NextTokenUi32()
    {
        return FromString<ui32>(NextToken());
    }

    float NextTokenFloat()
    {
        return FromString<float>(NextToken());
    }

    bool Eof()
    {
        return m_reader.Eof();
    }
};

typedef vector<float> TFloatVector;

struct TPicture
{
    static const size_t SIZE = 28;
    static const size_t SIZE2MUL = 4;
    static const size_t SIZE2 = SIZE/SIZE2MUL;
    // static const size_t VECTOR_SIZE = SIZE*SIZE;
    static const size_t VECTOR_SIZE = SIZE*SIZE + 2*SIZE2*SIZE2 + 1 + 10;

    typedef vector<TUi8Data> TUi8Matrix;
    TUi8Matrix m_matrix;
    TFloatVector m_features;
    int m_digit;

    TPicture(const TUi8Data& data, bool test)
    {
        if (!test)
        {
            m_digit = data[0];
        }
        else
        {
            m_digit = -1;
        }

        const int offset = (test) ? 0 : 1;

        if (data.size() != SIZE*SIZE + offset)
        {
            throw TException("bad data size '" + std::to_string(data.size()) + "'");
        }

        {
            TUi8Data dummy(SIZE);
            m_matrix.resize(SIZE, dummy);
        }

        for (size_t i = 0; i < m_matrix.size(); ++i)
        {
            for (size_t j = 0; j < m_matrix[i].size(); ++j)
            {
                m_matrix[i][j] = data[i*SIZE + j + offset];
            }
        }

        CalcFeatures();
    }

    void CalcFeatures()
    {
        m_features.resize(VECTOR_SIZE);
        size_t index = 0;
        float sum = 0;
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                sum += m_matrix[i][j];
                m_features[index++] = static_cast<float>(m_matrix[i][j])/256.f;
            }
        }
        m_features[index++] = sum/SIZE/SIZE/256.f;

        for (size_t i = 0; i < SIZE2; ++i)
        {
            for (size_t j = 0; j < SIZE2; ++j)
            {
                size_t sum = 0;
                for (size_t x = 0; x < SIZE2MUL; ++x)
                {
                    for (size_t y = 0; y < SIZE2MUL; ++y)
                    {
                        sum += m_matrix[4*i + x][4*j + y];
                    }
                }
                m_features[index++] = logf(1.f + (float)sum)/logf(1.f + 16.f*256.f);
            }
        }

        TUi8Matrix backup = m_matrix;
        Crop();
        for (size_t i = 0; i < SIZE2; ++i)
        {
            for (size_t j = 0; j < SIZE2; ++j)
            {
                size_t sum = 0;
                for (size_t x = 0; x < SIZE2MUL; ++x)
                {
                    for (size_t y = 0; y < SIZE2MUL; ++y)
                    {
                        sum += m_matrix[4*i + x][4*j + y];
                    }
                }
                m_features[index++] = logf(1.f + (float)sum)/logf(1.f + 16.f*256.f);
            }
        }

        m_matrix = backup;

        size_t components = 0;
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                if (m_matrix[i][j] == 0)
                {
                    Fill(i, j);
                    ++components;
                }
            }
        }
        if (components > 9)
        {
            components = 9;
        }
        m_features[index + components] = 1;
        index += 10;

        m_matrix.swap(backup);
    }

    int Digit() const
    {
        if (-1 != m_digit)
        {
            return m_digit;
        }
        else
        {
            throw TException("Digit() only for test data");
        }
    }

    void Fill(int x, int y)
    {
        if (x < 0 || x >= SIZE)
            return;
        if (y < 0 || y >= SIZE)
            return;
        if (0 == m_matrix[x][y])
        {
            m_matrix[x][y] = -1;
            static const int DIRS[] = {1, 0, -1, 0, 0, 1, 0, -1};
            for (size_t i = 0; i < 4; ++i)
                Fill(x + DIRS[2*i], y + DIRS[2*i + 1]);
        }
    }

    ui8 Get(size_t i, size_t j) const
    {
        return m_matrix[i][j];
    }

    ui8 GetDigit(size_t i, size_t j) const
    {
        ui8 value = Get(i, j);
        if (value)
        {
            return 1 + ((int)value*9)/256;
        }
        else
        {
            return 0;
        }
    }

    void Draw(FILE* fOut = stdout) const
    {
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                fprintf(fOut, "%d", GetDigit(i, j));
            }
            fprintf(fOut, "\n");
        }
        fprintf(fOut, "\n");
    }

    void DrawLine(FILE* fOut = stdout) const
    {
        fprintf(fOut, "-1");
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                fprintf(fOut, ",%d", Get(i, j));
            }
        }
        fprintf(fOut, "\n");
    }

    void Write(TFileWriter& fw) const
    {
        DrawLine(fw.GetHandle());
    }

    static bool IsZero(const TUi8Data& data)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i])
            {
                return false;
            }
        }
        return true;
    }

    TUi8Data Line(size_t index) const
    {
        return m_matrix[index];
    }

    TUi8Data Column(size_t index) const
    {
        TUi8Data result(SIZE);
        for (size_t i = 0; i < SIZE; ++i)
        {
            result[i] = m_matrix[i][index];
        }
        return result;
    }

    void Crop()
    {
        m_matrix[SIZE - 1][SIZE - 1] = 0;
        while (IsZero(Line(0)))
        {
            TUi8Data erase = *m_matrix.begin();
            m_matrix.erase(m_matrix.begin());
            m_matrix.push_back(erase);
        }
        while (IsZero(Column(0)))
        {
            for (size_t i = 0; i < SIZE; ++i)
            {
                TUi8Data& line = m_matrix[i];
                line.erase(line.begin());
                line.push_back(0);
            }
        }
    }

    const TFloatVector& AsVector() const
    {
        return m_features;
    }

    void SaveBMP(const string& s) const
    {
        BMP bmp;
        bmp.SetSize(SIZE, SIZE);
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                RGBApixel pixel;
                pixel.Blue = 255 - Get(i, j);
                pixel.Red = 255 - Get(i, j);
                pixel.Green = 255 - Get(i, j);
                pixel.Alpha = 0;
                bmp.SetPixel(j, i, pixel);
            }
        }
        if (!bmp.WriteToFile(s.c_str()))
        {
            throw TException("Could not SaveBMP '" + s + "'");
        }
    }
};

typedef vector<TPicture> TPictures;

void SplitLearnTest(const TRows& input, float ratio, TRows* learn, TRows* test)
{
    TTimer timer("SplitTestLearn");
    learn->clear();
    test->clear();
    for (TRows::const_iterator toRow = input.begin(); toRow != input.end(); ++toRow)
    {
        if (Rand01() < ratio)
        {
            learn->push_back(*toRow);
        }
        else
        {
            test->push_back(*toRow);
        }
    }
}

struct IProbEstimator
{
    virtual float Estimate(const TPicture& picture) const = 0;

    virtual ~IProbEstimator()
    {
    }
};

typedef vector<float> TFloatVector;
typedef vector<TFloatVector> TFloatMatrix;

template<typename T>
T Sqr(T x)
{
    return x*x;
}

struct TCosineEstimator : public IProbEstimator
{
    struct TWeights : public TFloatMatrix
    {
        TWeights()
        {
            Init();
        }

        TWeights(const TPicture& picture)
        {
            Init();
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    (*this)[i][j] = picture.Get(i, j);
                }
            }
        }

        void Init()
        {
            TFloatVector dummy(TPicture::SIZE);
            resize(TPicture::SIZE, dummy);
        }

        void Add(const TPicture& picture)
        {
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    (*this)[i][j] += picture.Get(i, j);
                }
            }
        }

        void Normalize()
        {
            float normalizer = 0.f;
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    normalizer += Sqr((*this)[i][j]);
                }
            }
            normalizer = sqrtf(normalizer);
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    (*this)[i][j] /= normalizer;
                }
            }
        }

        float Mul(const TWeights& w) const
        {
            float result = 0.f;
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    result += (*this)[i][j]*w[i][j];
                }
            }
            return result;
        }
    };

    TWeights m_weights;

    TCosineEstimator()
    {
    }

    void Learn(const TPicture& picture)
    {
        m_weights.Add(picture);
    }

    void EndLearn()
    {
        m_weights.Normalize();
    }

    virtual float Estimate(const TPicture& picture) const
    {
        TWeights pictureWeights(picture);
        pictureWeights.Normalize();
        return m_weights.Mul(pictureWeights);
    }
};

typedef pair<size_t, float> TBest;

typedef vector<IProbEstimator*> IProbEstimators;

TBest Choose(IProbEstimators estimators, const TPicture& picture, const string& name, size_t index)
{
    float best = 0.f;
    float nextToBest = 0.f;
    size_t bestIndex = 0;
    vector<float> probes(estimators.size());
    for (size_t i = 0; i < estimators.size(); ++i)
    {
        const float prob = estimators[i]->Estimate(picture);
        probes[i] = prob;
        if (prob > best)
        {
            nextToBest = best;
            best = prob;
            bestIndex = i;
        }
    }
    if ((best < 0.8f) || (nextToBest + 0.2f > best))
    {
        MkDir(name);
        picture.SaveBMP(name + "/" + std::to_string(index) + ".bmp");
        TFileWriter fOut(name + "/" + std::to_string(index) + ".txt");
        fOut.Write(std::to_string(bestIndex) + "\t" + std::to_string(best) + "\t" + std::to_string(nextToBest) + "\n");
        for (size_t i = 0; i < probes.size(); ++i)
        {
            fOut.Write( std::to_string(i) + "\t" + std::to_string(probes[i]) + "\n" );
        }
        picture.Write(fOut);
    }
    return make_pair(bestIndex, best);
}

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
        typedef vector<TSinapse> TSinapses;
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
        typedef vector<TInvertedSinapse> TInvertedSinapses;
        TInvertedSinapses m_invertedSinapses;

        void AddSinapse(ui16 inputIndex, float weight = 0.f)
        {
            weight *= 10.f;
            if (0.f == weight)
            {
                weight = (Rand01() - 0.5f)*2.f;
            }
            m_sinapses.push_back( TSinapse(inputIndex, weight) );
        }

        void Save(TFileWriter& fOut) const
        {
            fOut.Write(std::to_string(m_sinapses.size()) + "\n");
            for (size_t i = 0; i < m_sinapses.size(); ++i)
            {
                fOut.Write("\t" + std::to_string(m_sinapses[i].m_weight) + "\n");
            }
        }

        void Load(TTokenReader& fIn)
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
    };
    typedef vector<TNeuron> TNeurons;
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

    static float Sigmoid(float value)
    {
        return 1.f / (1.f + expf(-value));
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

    void CalculateValues(const TFloatVector& input, TFloatVector* result) const
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

    float GetOutput(const TFloatVector& input) const
    {
        TFloatVector data;
        CalculateValues(input, &data);
        return data.back();
    }

    float Estimate(const TPicture& input) const
    {
        return GetOutput(input.AsVector());
    }

    void Inflate()
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
        const float learnRate = 0.5f/sqrtf(iteration + 1);
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

    void Save(const std::string& filename) const
    {
        TFileWriter fOut(filename);
        fOut.Write( std::to_string(m_neurons.size()) + "\n" );
        for (size_t i = 0; i < m_neurons.size(); ++i)
        {
            m_neurons[i].Save(fOut);
        }
    }

    void Load(const std::string& filename)
    {
        TTokenReader fIn(filename);
        ui32 len = fIn.NextTokenUi32();
        if (len > m_neurons.size())
        {
            throw TException("bad neuro net input");
        }
        for (ui32 i = 0; i < len; ++i)
        {
            m_neurons[i].Load(fIn);
        }
    }
};

TEST(NeuralNet, XOR)
{
    TNeuralEstimator estimator;
    {
        estimator.SetInputSize(2);
        size_t prevLayerBegin = 0;
        size_t prevLayerSize = 2;
        for (size_t iLayer = 0; iLayer < 1; ++iLayer)
        {
            size_t layerBegin = estimator.Size();
            for (size_t i = 0; i < 3; ++i)
            {
                TNeuralEstimator::TNeuron neuron;
                for (size_t j = 0; j < prevLayerSize; ++j)
                {
                    neuron.AddSinapse(prevLayerBegin + j);
                }
                estimator.Add(neuron);
            }
            prevLayerBegin = layerBegin;
            prevLayerSize = 3;
        }
        TNeuralEstimator::TNeuron neuronOutput;
        for (size_t j = 0; j < prevLayerSize; ++j)
        {
            neuronOutput.AddSinapse(prevLayerBegin + j);
        }
        estimator.Add(neuronOutput);
        estimator.Prepare();
    }

    {
        for (size_t iLearnIt = 0; iLearnIt < 100000; ++iLearnIt)
        {
            {
                estimator.Inflate();
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        estimator.BackPropagation(input, result, iLearnIt);
                    }
                }
            }
            {
                float error = 0.f;
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        const float netResult = estimator.GetOutput(input);
                        // printf("%d %d %f %f\n", (int)x, (int)y, result, netResult);
                        error += Sqr(result - netResult);
                    }
                }
                // printf("Error %d: %f\n", iLearnIt, error);
            }
        }
    }
}

TEST(NeuralNet, SaveLoad)
{
    TNeuralEstimator estimator;
    {
        estimator.SetInputSize(2);
        size_t prevLayerBegin = 0;
        size_t prevLayerSize = 2;
        for (size_t iLayer = 0; iLayer < 1; ++iLayer)
        {
            size_t layerBegin = estimator.Size();
            for (size_t i = 0; i < 3; ++i)
            {
                TNeuralEstimator::TNeuron neuron;
                for (size_t j = 0; j < prevLayerSize; ++j)
                {
                    neuron.AddSinapse(prevLayerBegin + j);
                }
                estimator.Add(neuron);
            }
            prevLayerBegin = layerBegin;
            prevLayerSize = 3;
        }
        TNeuralEstimator::TNeuron neuronOutput;
        for (size_t j = 0; j < prevLayerSize; ++j)
        {
            neuronOutput.AddSinapse(prevLayerBegin + j);
        }
        estimator.Add(neuronOutput);
        estimator.Prepare();
    }

    {
        for (size_t iLearnIt = 0; iLearnIt < 10; ++iLearnIt)
        {
            {
                estimator.Inflate();
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        estimator.BackPropagation(input, result, iLearnIt);
                    }
                }
            }
            {
                float error = 0.f;
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        const float netResult = estimator.GetOutput(input);
                        error += Sqr(result - netResult);
                    }
                }
            }
        }
    }

    {
        float error = 0.f;
        for (size_t x = 0; x < 2; ++x)
        {
            for (size_t y = 0; y < 2; ++y)
            {
                TFloatVector input;
                input.push_back(x);
                input.push_back(y);
                const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                const float netResult = estimator.GetOutput(input);
                error += Sqr(result - netResult);
            }
        }
        estimator.Save("test.weights");
        estimator.Load("test.weights");
        for (size_t x = 0; x < 2; ++x)
        {
            for (size_t y = 0; y < 2; ++y)
            {
                TFloatVector input;
                input.push_back(x);
                input.push_back(y);
                const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                const float netResult = estimator.GetOutput(input);
                error -= Sqr(result - netResult);
            }
        }
        EXPECT_TRUE(fabs(error) < 0.000001f);
    }
}

template<typename T>
void Shuffle(vector<T>& v)
{
    for (ssize_t i = static_cast<ssize_t>(v.size()) - 1; i >= 1; --i)
    {
        size_t index = rand() % i;
        swap(v[index], v[i]);
    }
}

void MakePictures(const TRows& rows, TPictures* pictures, bool test)
{
    TTimer timerLearn("MakePictures " + std::to_string(rows.size()));
    pictures->clear();
    for (size_t i = 0; i < rows.size(); ++i)
    {
        pictures->push_back( TPicture(rows[i], test) );
    }
}

int main(int argc, char* argv[])
{
    TCommandLineParser parser(argc, argv);
    const bool unittests = parser.Has('u', "unittests", "run unittests");
    const bool draw = parser.Has('d', "draw", "draw train");
    const bool verbose = parser.Has('v', "verbose", "verbose");
    const bool cosine = parser.Has('c', "cosine", "cosine");
    const bool neural = parser.Has('n', "neural", "neural");
    const int limit = parser.GetInt('l', "limit", "limit input", std::numeric_limits<int>::max());
    const string loadFrom = parser.Get('L', "load", "start from NN", "");
    parser.AutoUsage();

    if (draw)
    {
        TCSVReader trainData("train.csv", true);
        for (size_t i = 0; i < trainData.m_rows.size(); ++i)
        {
            TPicture(trainData.m_rows[i], false).Draw();
            printf("\n");
        }
    }
    else if (unittests)
    {
        InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    else if (cosine)
    {
        TCSVReader trainData("train.csv", true);
        {
            TRows learn;
            TRows test;
            SplitLearnTest(trainData.m_rows, 0.9, &learn, &test);

            TCosineEstimator estimators[10];
            {
                TTimer timerLearn("Learn");
                for (size_t i = 0; i < learn.size(); ++i)
                {
                    TPicture p(learn[i], false);
                    // p.Crop();
                    estimators[learn[i][0]].Learn(p);
                }
            }

            {
                IProbEstimators pEstimators(10);
                for (size_t i = 0; i < 10; ++i)
                {
                    pEstimators[i] = &estimators[i];
                }

                TTimer timerLearn("Test");
                size_t preceision = 0;
                for (size_t i = 0; i < test.size(); ++i)
                {
                    TPicture p(test[i], false);
                    // p.Crop();
                    TBest best = Choose(pEstimators, p, "test", i);
                    if (verbose)
                    {
                        printf("%d %d\n", (int)test[i][0], best.first);
                        p.Draw();
                    }
                    if (test[i][0] == best.first)
                    {
                        ++preceision;
                    }
                }
                printf("Precision: %f\n", ((float)preceision)/test.size());
            }
        }

        {
            TCosineEstimator estimators[10];
            {
                TTimer timerLearn("Learn");
                for (size_t i = 0; i < trainData.m_rows.size(); ++i)
                {
                    TPicture p(trainData.m_rows[i], false);
                    estimators[trainData.m_rows[i][0]].Learn(p);
                }
            }

            {
                IProbEstimators pEstimators(10);
                for (size_t i = 0; i < 10; ++i)
                {
                    pEstimators[i] = &estimators[i];
                }

                TTimer timerLearn("Test");
                TCSVReader testData("test.csv", true);
                TCSVWriter writer("cosine.csv");
                for (size_t i = 0; i < testData.m_rows.size(); ++i)
                {
                    TPicture p(testData.m_rows[i], true);
                    TBest best = Choose(pEstimators, p, "test", i);
                    writer.Put( std::to_string(best.first) );
                    writer.NewLine();
                    if (verbose)
                    {
                        printf("%d\n", best.first);
                        p.Draw();
                    }
                }
            }
        }
    }
    else if (neural)
    {
        TPictures pLearn;
        {
            TCSVReader trainData("train.csv", true, limit);
            MakePictures(trainData.m_rows, &pLearn, false);
        }
        TPictures pTest;
        {
            TCSVReader testData("test.csv", true);
            MakePictures(testData.m_rows, &pTest, true);
        }
        {
            TPictures pHand;
            {
                TCSVReader handData("hand.csv", true);
                MakePictures(handData.m_rows, &pHand, false);
            }
            pLearn.insert(pLearn.end(), pHand.begin(), pHand.end());
        }
        printf("Train=%d, Test=%d\n", static_cast<int>(pLearn.size()), static_cast<int>(pTest.size()));
        vector<TNeuralEstimator> estimators(10);
        {
            TTimer timerLearn("Configure neural net");
            for (size_t digit = 0; digit < 10; ++digit)
            {
                TNeuralEstimator& estimator = estimators[digit];

                estimator.SetInputSize(TPicture::VECTOR_SIZE);
                size_t prevLayerBegin = 0;
                size_t prevLayerSize = TPicture::VECTOR_SIZE;
                for (size_t i = 0; i <= 2; ++i)
                {
                    const size_t layerBegin = estimator.Size();
                    const size_t layerSize = (i == 2) ? 1 : ( (i == 0) ? TPicture::VECTOR_SIZE : 100 );
                    for (size_t k = 0; k < layerSize; ++k)
                    {
                        TNeuralEstimator::TNeuron neuron;
                        for (size_t j = 0; j < prevLayerSize; ++j)
                        {
                            neuron.AddSinapse(prevLayerBegin + j, ( (k + j + i) % TPicture::VECTOR_SIZE ) ? 0.f : 1.f);
                        }
                        estimator.Add(neuron);
                    }
                    prevLayerBegin = layerBegin;
                    prevLayerSize = layerSize;
                }
                estimator.Prepare();
                if (!loadFrom.empty())
                {
                    estimator.Load(loadFrom + std::to_string(digit));
                }
            }
        }

        {
            TTimer timerLearn("Learn");
            for (size_t iLearnIt = 0; iLearnIt < 100; ++iLearnIt)
            {
                float ratio;
                switch (iLearnIt)
                {
                    case 0:
                        ratio = 0.01f;
                        break;
                    case 1:
                        ratio = 0.05f;
                        break;
                    case 2:
                        ratio = 0.2f;
                        break;
                    case 3:
                        ratio = 0.5f;
                        break;
                    default:
                        ratio = 1.1f;
                }
                const string name = "debug/learn" + std::to_string(iLearnIt);
                MkDir(name);
                {
                    TTimer timerLearn("Learn iteration " + std::to_string(iLearnIt) + " " + std::to_string(ratio));
                    for (size_t digit = 0; digit < 10; ++digit)
                    {
                        TNeuralEstimator& estimator = estimators[digit];
                        TTimer timerLearn2("Learn iteration " + std::to_string(iLearnIt) + " digit " + std::to_string(digit));
                        estimator.Inflate();

                        vector<size_t> indexes(pLearn.size());
                        for (size_t i = 0; i < indexes.size(); ++i)
                        {
                            indexes[i] = i;
                        }
                        Shuffle(indexes);

                        for (size_t i = 0; i < pLearn.size(); ++i)
                        {
                            if (Rand01() > ratio)
                            {
                                continue;
                            }

                            const TPicture& p = pLearn[indexes[i]];
                            const float value = (p.Digit() == digit) ? 1.f : 0.f;
                            if (!(i % 100))
                            {
                                printf("before %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                            }
                            estimator.BackPropagation(p.AsVector(), value, iLearnIt);
                            if (!(i % 100))
                            {
                                printf("after %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                                fflush(stdout);
                            }
                        }
                        estimator.Save(name + "/NN" + std::to_string(digit));
                    }
                }
                {
                    TTimer timerTest("Test iteration " + std::to_string(iLearnIt));
                    for (size_t digit = 0; digit < 10; ++digit)
                    {
                        float error = 0.f;
                        size_t num = 0;
                        for (size_t i = 0; i < pLearn.size(); ++i)
                        {
                            if (Rand01() > 2.f*ratio)
                            {
                                continue;
                            }

                            const TPicture& p = pLearn[i];

                            if (i && !(i % 4000))
                            {
                                printf("%.2f\n", ((float)i)/pLearn.size());
                                fflush(stdout);
                            }

                            const float result = (p.Digit() == digit) ? 1.f : 0.f;
                            const float netResult = estimators[digit].GetOutput(p.AsVector());
                            const float dError = Sqr(result - netResult);
                            if (dError > 0.25f)
                            {
                                p.SaveBMP(name + "/" + std::to_string(i) + ".bmp");
                                TFileWriter fOut(name + "/" + std::to_string(i) + "_" + std::to_string(digit) + ".txt");
                                fOut.Write( std::to_string(i) + "\t" + std::to_string(p.Digit()) + "\t" + std::to_string(digit) + "\t" + std::to_string(result) + "\t" + std::to_string(netResult) + "\n" );
                                p.Write(fOut);
                            }
                            error += dError;
                            ++num;
                        }
                        printf("Error %d %d: %f %f\n", (int)iLearnIt, (int)digit, error, error/num);
                    }
                }
                const string testName = "debug/testNN" + std::to_string(iLearnIt);
                {
                    TTimer timerApply("Apply " + std::to_string(iLearnIt));
                    TCSVWriter writer("neural.csv");
                    IProbEstimators pEstimators(10);
                    for (size_t i = 0; i < 10; ++i)
                    {
                        pEstimators[i] = &estimators[i];
                    }

                    TFileWriter fOut("dump.txt");
                    for (size_t i = 0; i < pTest.size(); ++i)
                    {
                        if (!(i % 500))
                        {
                            printf("%f\n", ((float)i)/pTest.size());
                            fflush(stdout);
                        }

                        const TPicture& p = pTest[i];
                        TBest best = Choose(pEstimators, p, testName, i);
                        writer.Put( std::to_string(best.first) );
                        writer.NewLine();
                        if (verbose)
                        {
                            printf("%d\n", best.first);
                            p.Draw();
                        }

                        for (size_t j = 0; j < 10; ++j)
                        {
                            fOut.Write(std::to_string(i) + "\t" + std::to_string(j) + "\t" + std::to_string(estimators[j].Estimate(p)) + "\n");
                        }
                        fOut.Write(std::to_string(i) + "\t" + std::to_string(best.first) + "\t" + std::to_string(best.second) + "\n");
                        p.Draw(fOut.GetHandle());
                        p.DrawLine(fOut.GetHandle());
                    }
                }
            }
        }
    }
    else
    {
        TPictures pLearn;
        TPictures pTest;
        {
            TCSVReader trainData("train.csv", true, limit);
            TRows learn;
            TRows test;
            SplitLearnTest(trainData.m_rows, 0.9, &learn, &test);
            MakePictures(learn, &pLearn, false);
            MakePictures(test, &pTest, false);
        }

        TNeuralEstimator estimator;
        {
            TTimer timerLearn("Configure neural net");
            estimator.SetInputSize(TPicture::VECTOR_SIZE);
            size_t prevLayerBegin = 0;
            size_t prevLayerSize = TPicture::VECTOR_SIZE;
            for (size_t i = 0; i < 2; ++i)
            {
                const size_t layerBegin = estimator.Size();
                const size_t layerSize = (i == 0) ? TPicture::VECTOR_SIZE : 100;
                for (size_t k = 0; k < layerSize; ++k)
                {
                    TNeuralEstimator::TNeuron neuron;
                    for (size_t j = 0; j < prevLayerSize; ++j)
                    {
                        neuron.AddSinapse(prevLayerBegin + j);
                    }
                    estimator.Add(neuron);
                }
                prevLayerBegin = layerBegin;
                prevLayerSize = layerSize;
            }
            {
                TNeuralEstimator::TNeuron neuronOutput;
                for (size_t j = 0; j < prevLayerSize; ++j)
                {
                    neuronOutput.AddSinapse(prevLayerBegin + j);
                }
                estimator.Add(neuronOutput);
            }
            estimator.Prepare();
        }

        {
            TTimer timerLearn("Learn " + std::to_string(pLearn.size()));
            for (size_t iLearnIt = 0; iLearnIt < 100; ++iLearnIt)
            {
                {
                    TTimer timerLearn("Learn iteration " + std::to_string(iLearnIt));
                    estimator.Inflate();
                    for (size_t i = 0; i < pLearn.size(); ++i)
                    {
                        const TPicture& p = pLearn[i];
                        const float value = (p.Digit() == 0) ? 1.f : 0.f;
                        if (!(i % 100))
                        {
                            printf("before %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                        }
                        estimator.BackPropagation(p.AsVector(), value, iLearnIt);
                        if (!(i % 100))
                        {
                            printf("after %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                        }
                    }
                }
                {
                    TTimer timerLearn("Test iteration " + std::to_string(iLearnIt));
                    float error = 0.f;
                    for (size_t i = 0; i < pTest.size(); ++i)
                    {
                        const TPicture& p = pTest[i];
                        const float result = (p.Digit() == 0) ? 1.f : 0.f;
                        const float netResult = estimator.GetOutput(p.AsVector());
                        error += Sqr(result - netResult);
                    }
                    printf("Error %d: %f %f\n", iLearnIt, error, error/pTest.size());
                }
            }
        }
    }

    return 0;
}
