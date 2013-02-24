#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <vector>

#include "gtest/gtest.h"

using namespace std;
using namespace testing;

typedef unsigned char ui8;
static_assert(1 == sizeof(ui8), "ui8 bad size");
typedef unsigned short ui16;
static_assert(2 == sizeof(ui16), "ui16 bad size");

struct TException : public exception
{
    string m_message;

    TException()
    {
    }

    TException(const TException& e)
        : m_message(e.m_message)
    {
    }

    TException(const string& message)
        : m_message(message)
    {
    }

    virtual const char* what() const noexcept
    {
        return m_message.c_str();
    }

    ~TException() noexcept
    {
    }
};

struct TFileReader
{
    FILE* m_file;

    TFileReader(const string& filename)
    {
        m_file = fopen(filename.c_str(), "rb");
        if (!m_file)
        {
            throw TException((string("could not open '") + filename + "'").c_str());
        }
    }

    bool ReadLine(string* result)
    {
        const static size_t BUFFER_LEN = 16536;
        char buffer[BUFFER_LEN];
        if (fgets(buffer, BUFFER_LEN, m_file))
        {
            *result = buffer;
            return true;
        }
        else
        {
            return false;
        }
    }

    ~TFileReader()
    {
        fclose(m_file);
    }
};

struct TFileWriter
{
    FILE* m_file;

    TFileWriter(const string& filename)
    {
        m_file = fopen(filename.c_str(), "wb");
    }

    void Write(const std::string& s)
    {
        if (fwrite(s.c_str(), s.length(), 1, m_file) != 1)
        {
            throw TException("write failed");
        }
    }

    FILE* GetHandle()
    {
        return m_file;
    }

    ~TFileWriter()
    {
        fclose(m_file);
    }
};

struct TTimer
{
    string m_message;
    clock_t m_begin;

    TTimer(const string& message)
        : m_message(message)
    {
        m_begin = clock();
        fprintf(stderr, "============================== %s begin ==============================\n", m_message.c_str());
    }

    ~TTimer()
    {
        clock_t end = clock();
        clock_t diff = end - m_begin;
        fprintf(stderr, "============================== %s end - %d %d ==============================\n", m_message.c_str(), (int)(diff / CLOCKS_PER_SEC), (int)(diff % CLOCKS_PER_SEC));
    }
};

typedef vector<string> TStringVector;

void Split(const std::string& line, char sep, TStringVector* result)
{
    result->clear();
    if (!line.empty())
    {
        std::string::const_iterator begin = line.begin();
        std::string::const_iterator now = line.begin();

        while (now <= line.end())
        {
            if (*now == sep || *now == 0)
            {
                if (begin != now)
                {
                    result->push_back(string(begin, now));
                }
                begin = now + 1;
            }
            ++now;
        }
    }
}

TEST(Split, Basics)
{
    TStringVector sv;
    Split("a,b,c", ',', &sv);
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], "a");
    EXPECT_EQ(sv[1], "b");
    EXPECT_EQ(sv[2], "c");
}

template<typename T>
T FromString(const std::string& s)
{
    T::Unimplemented;
}

template<>
ui8 FromString<ui8>(const std::string& s)
{
    ui8 result = 0;
    if (!s.empty())
    {
        for (size_t i = 0; i < s.length(); ++i)
            result = 10*result + s[i] - '0';
        return result;
    }
    else
    {
        throw TException("empty string");
    }
}

template<>
int FromString<int>(const std::string& s)
{
    int result = 0;
    if (!s.empty())
    {
        for (size_t i = 0; i < s.length(); ++i)
            result = 10*result + s[i] - '0';
        return result;
    }
    else
    {
        throw TException("empty string");
    }
}

template<typename T>
std::string ToString(const T&)
{
    T::Unimplemented;
}

template<>
std::string ToString<size_t>(const size_t& value)
{
    static const size_t BUFFER_SIZE = 64;
    char buffer[BUFFER_SIZE];
    if (snprintf(buffer, BUFFER_SIZE, "%u", (unsigned int)value) < 0)
    {
        throw TException("ToString failed");
    }
    return buffer;
}

template<>
std::string ToString<int>(const int& value)
{
    static const size_t BUFFER_SIZE = 64;
    char buffer[BUFFER_SIZE];
    if (snprintf(buffer, BUFFER_SIZE, "%d", value) < 0)
    {
        throw TException("ToString failed");
    }
    return buffer;
}

template<>
std::string ToString<float>(const float& value)
{
    static const size_t BUFFER_SIZE = 64;
    char buffer[BUFFER_SIZE];
    if (snprintf(buffer, BUFFER_SIZE, "%f", value) < 0)
    {
        throw TException("ToString failed");
    }
    return buffer;
}

typedef vector<ui8> TUi8Data;
typedef vector<TUi8Data> TRows;

struct TCSVReader
{
    TRows m_rows;
    TFileReader m_fileReader;

    TCSVReader(const string& filename, bool verbose, size_t limit = std::numeric_limits<size_t>::max())
        : m_fileReader(filename)
    {
        TTimer timer("CVSReader '" + filename + "' " + ToString(limit));
        string line;
        if (m_fileReader.ReadLine(&line))
        {
            while (m_fileReader.ReadLine(&line) && (m_rows.size() < limit))
            {
                vector<string> tokens;
                Split(line, ',', &tokens);
                m_rows.push_back( TUi8Data() );
                m_rows.back().resize(tokens.size());
                for (size_t i = 0; i < tokens.size(); ++i)
                {
                    m_rows.back()[i] = FromString<ui8>(tokens[i]);
                }
            }
        }
    }
};

struct TCSVWriter
{
    TFileWriter m_fileWriter;
    bool m_first;

    TCSVWriter(const string& filename)
        : m_fileWriter(filename)
        , m_first(true)
    {
    }

    void NewLine()
    {
        m_fileWriter.Write("\n");
        m_first = true;
    }

    void Put(const string& s)
    {
        if (!m_first)
        {
            m_fileWriter.Write(",");
        }
        m_first = false;
        m_fileWriter.Write(s);
    }
};

typedef vector<float> TFloatVector;

struct TPicture
{
    static const size_t SIZE = 28;
    static const size_t SIZE2 = 7;
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
            throw TException("bad data size '" + ToString(data.size()) + "'");
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
                for (size_t x = 0; x < 4; ++x)
                {
                    for (size_t y = 0; y < 4; ++y)
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
                for (size_t x = 0; x < 4; ++x)
                {
                    for (size_t y = 0; y < 4; ++y)
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
                fprintf(fOut, ",%d", GetDigit(i, j));
            }
        }
        fprintf(fOut, "\n");
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
};

typedef vector<TPicture> TPictures;

struct TCommandLineParser
{
    struct TOption
    {
        char m_option;
        string m_longOption;
        string m_description;
        bool m_isInt;
        int m_intDefault;

        TOption()
        {
        }

        TOption(char option, const string& longOption, const string& description, bool isInt, int intDefault)
            : m_option(option)
            , m_longOption(longOption)
            , m_description(description)
            , m_isInt(isInt)
            , m_intDefault(intDefault)
        {
        }
    };
    typedef vector<TOption> TOptions;

    TOptions m_options;
    TStringVector m_args;
    bool m_error;
    string m_strError;

    TCommandLineParser(int argc, char* const argv[])
        : m_error(false)
    {
        m_args.resize(argc);
        for (size_t i = 0; i < argc; ++i)
        {
            m_args[i] = argv[i];
        }
    }

    bool Has(char option, const string& longOption, const string& description)
    {
        m_options.push_back( TOption(option, longOption, description, false, 0) );

        string key = "-";
        key += option;
        string longKey = "--";
        longKey += longOption;
        for (size_t i = 0; i < m_args.size(); ++i)
        {
            if (m_args[i] == key || m_args[i] == longKey)
            {
                return true;
            }
        }

        return false;
    }

    int GetInt(char option, const string& longOption, const string& description, int defaultValue)
    {
        m_options.push_back( TOption(option, longOption, description, true, defaultValue) );

        string key = "-";
        key += option;
        string longKey = "--";
        longKey += longOption;
        for (size_t i = 0; i < m_args.size(); ++i)
        {
            if (m_args[i] == key || m_args[i] == longKey)
            {
                if (i + 1 < m_args.size())
                {
                    try
                    {
                        return FromString<int>(m_args[i + 1]);
                    }
                    catch (...)
                    {
                        m_error = true;
                        m_strError = "cannot cast to integer '" + m_args[i + 1] + "'";
                    }
                }
                else
                {
                    m_error = true;
                    m_strError = "not enought arguments";
                }
                return true;
            }
        }

        return defaultValue;
    }

    bool AutoUsage()
    {
        if (m_error || Has('?', "--help", "print usage help"))
        {
            for (size_t i = 0; i < m_options.size(); ++i)
            {
                const TOption& option = m_options[i];
                printf("-%c (--%s) - %s", option.m_option, option.m_longOption.c_str(), option.m_description.c_str());
                if (option.m_isInt)
                {
                    printf(" [int, default=%d]", option.m_intDefault);
                }
                printf("\n");
            }
            printf("\n");
            exit(1);
        }

        if (m_error)
        {
            fprintf(stderr, "argument parsing problem: %s\n", m_strError.c_str());
            throw TException("argument parsing problem: " + m_strError + "\n");
        }
    }
};

float Rand01()
{
    return ((float)rand())/RAND_MAX;
}

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

TBest Choose(IProbEstimators estimators, const TPicture& picture)
{
    float best = 0.f;
    float bestIndex = 0;
    for (size_t i = 0; i < estimators.size(); ++i)
    {
        float prob = estimators[i]->Estimate(picture);
        if (prob > best)
        {
            best = prob;
            bestIndex = i;
        }
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

        void AddSinapse(ui16 inputIndex)
        {
            m_sinapses.push_back( TSinapse(inputIndex, (Rand01() - 0.5f)*0.1f) );
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
                s.m_weight += learnRate*delta[i]*data[s.m_index];
            }
        }
    }

    size_t Size() const
    {
        return m_neurons.size();
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
        for (size_t iLearnIt = 0; iLearnIt < 1000; ++iLearnIt)
        {
            {
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
                        printf("%d %d %f %f\n", (int)x, (int)y, result, netResult);
                        error += Sqr(result - netResult);
                    }
                }
                printf("Error %d: %f\n", iLearnIt, error);
            }
        }
    }
}

void MakePictures(const TRows& rows, TPictures* pictures, bool test)
{
    TTimer timerLearn("MakePictures " + ToString(rows.size()));
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
                    TBest best = Choose(pEstimators, p);
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
                    TBest best = Choose(pEstimators, p);
                    writer.Put( ToString(best.first) );
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

        vector<TNeuralEstimator> estimators(10);
        {
            TTimer timerLearn("Configure neural net");
            for (size_t i = 0; i < 10; ++i)
            {
                TNeuralEstimator& estimator = estimators[i];

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
        }

        {
            TTimer timerLearn("Learn");
            for (size_t iLearnIt = 0; iLearnIt < 10; ++iLearnIt)
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
                {
                    TTimer timerLearn("Learn iteration " + ToString(iLearnIt));
                    for (size_t digit = 0; digit < 10; ++digit)
                    {
                        TNeuralEstimator& estimator = estimators[digit];
                        TTimer timerLearn2("Learn iteration " + ToString(iLearnIt) + " digit " + ToString(digit));
                        for (size_t i = 0; i < pLearn.size(); ++i)
                        {
                            const TPicture& p = pLearn[i];
                            if (Rand01() > ratio)
                            {
                                continue;
                            }
                            const float value = (pLearn[i].Digit() == digit) ? 1.f : 0.f;
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
                    }
                }
                {
                    TTimer timerTest("Test iteration " + ToString(iLearnIt));
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

                            const float result = (pLearn[i].Digit() == digit) ? 1.f : 0.f;
                            const float netResult = estimators[digit].GetOutput(p.AsVector());
                            error += Sqr(result - netResult);
                            ++num;
                        }
                        printf("Error %d %d: %f %f\n", (int)iLearnIt, (int)digit, error, error/num);
                    }
                }
                {
                    TTimer timerApply("Apply " + ToString(iLearnIt));
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
                        TBest best = Choose(pEstimators, p);
                        writer.Put( ToString(best.first) );
                        writer.NewLine();
                        if (verbose)
                        {
                            printf("%d\n", best.first);
                            p.Draw();
                        }

                        for (size_t j = 0; j < 10; ++j)
                        {
                            fOut.Write(ToString(i) + "\t" + ToString(j) + "\t" + ToString(estimators[j].Estimate(p)) + "\n");
                        }
                        fOut.Write(ToString(i) + "\t" + ToString(best.first) + "\t" + ToString(best.second) + "\n");
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
            TTimer timerLearn("Learn " + ToString(pLearn.size()));
            for (size_t iLearnIt = 0; iLearnIt < 100; ++iLearnIt)
            {
                {
                    TTimer timerLearn("Learn iteration " + ToString(iLearnIt));
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
                    TTimer timerLearn("Test iteration " + ToString(iLearnIt));
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
