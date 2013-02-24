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

typedef vector<ui8> TUi8Data;
typedef vector<TUi8Data> TRows;

struct TCSVReader
{
    TRows m_rows;
    TFileReader m_fileReader;

    TCSVReader(const string& filename, bool verbose, size_t limit = std::numeric_limits<size_t>::max())
        : m_fileReader(filename)
    {
        TTimer timer("CVSReader '" + filename + "'");
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

    typedef vector<TUi8Data> TUi8Matrix;
    TUi8Matrix m_matrix;

    TPicture(const TUi8Data& data, bool test)
    {
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

    void Draw()
    {
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                printf("%d", GetDigit(i, j));
            }
            printf("\n");
        }
        printf("\n");
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

    TFloatVector AsVector() const
    {
        TFloatVector result(SIZE*SIZE);
        size_t index = 0;
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                result[index++] = static_cast<float>(m_matrix[i][j])/256.f;
            }
        }
        return result;
    }
};

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

struct TNeuralEstimator
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
            m_sinapses.push_back( TSinapse(inputIndex, Rand01()/100.f) );
        }
    };
    typedef vector<TNeuron> TNeurons;
    TNeurons m_neurons;

    void SetInputSize(size_t inputSize)
    {
        m_inputSize = inputSize;
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
                m_neurons[sinapse.m_index].m_invertedSinapses.push_back( TNeuron::TInvertedSinapse(i, j) );
            }
        }
    }

    float GetOutput(const TFloatVector& input) const
    {
        assert( input.size() == m_inputSize );
        TFloatVector data(input);
        data.resize(Size());
        for (size_t i = 0; i < m_neurons.size(); ++i)
        {
            float value = 0.f;
            const TNeuron& neuron = m_neurons[i];
            for (size_t j = 0; j < neuron.m_sinapses.size(); ++j)
            {
                const TNeuron::TSinapse& sin = neuron.m_sinapses[j];
                value += sin.m_weight*data[sin.m_index];
            }
            data[i + m_inputSize] = Sigmoid(value);
        }
        return 0.f;
    }

    void BackPropagation(const TFloatVector& input, float targetOutput)
    {
        assert( input.size() == m_inputSize );
    }

    size_t Size() const
    {
        return m_inputSize + m_neurons.size();
    }
};

TEST(NeuralNet, XOR)
{
    TNeuralEstimator estimator;
    {
        TTimer timerLearn("Configure neural net");
        estimator.SetInputSize(2);
        for (size_t iLayer = 0; iLayer < 2; ++iLayer)
        {
            for (size_t i = 0; i < 2; ++i)
            {
                TNeuralEstimator::TNeuron neuron;
                for (size_t j = 0; j < 2; ++j)
                {
                    neuron.AddSinapse(estimator.Size() - 2);
                }
                estimator.Add(neuron);
            }
        }
        TNeuralEstimator::TNeuron neuronOutput;
        for (size_t j = 0; j < j; ++j)
        {
            neuronOutput.AddSinapse( estimator.Size() - Sqr(TPicture::SIZE) );
        }
        estimator.Add(neuronOutput);
        estimator.Prepare();
    }

    {
        for (size_t iLearnIt = 0; iLearnIt < 10; ++iLearnIt)
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
                        estimator.BackPropagation(input, result);
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
                printf("Error %d: %f\n", iLearnIt, error);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    TCommandLineParser parser(argc, argv);
    const bool unittests = parser.Has('u', "unittests", "run unittests");
    const bool draw = parser.Has('d', "draw", "draw train");
    const bool verbose = parser.Has('v', "verbose", "verbose");
    const bool cosine = parser.Has('c', "cosine", "cosine");
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
    else
    {
        TCSVReader trainData("train.csv", true, limit);
        {
            TRows learn;
            TRows test;
            SplitLearnTest(trainData.m_rows, 0.9, &learn, &test);

            TNeuralEstimator estimator;
            {
                TTimer timerLearn("Configure neural net");
                estimator.SetInputSize( Sqr(TPicture::SIZE) );
                for (size_t i = 0; i < 3; ++i)
                {
                    TNeuralEstimator::TNeuron neuron;
                    for (size_t j = 0; j < Sqr(TPicture::SIZE); ++j)
                    {
                        neuron.AddSinapse( estimator.Size() - Sqr(TPicture::SIZE) );
                    }
                    estimator.Add(neuron);
                }
                TNeuralEstimator::TNeuron neuronOutput;
                for (size_t j = 0; j < Sqr(TPicture::SIZE); ++j)
                {
                    neuronOutput.AddSinapse( estimator.Size() - Sqr(TPicture::SIZE) );
                }
                estimator.Add(neuronOutput);
                estimator.Prepare();
            }

            {
                TTimer timerLearn("Learn");
                for (size_t iLearnIt = 0; iLearnIt < 10; ++iLearnIt)
                {
                    {
                        TTimer timerLearn("Learn Iteration " + ToString(iLearnIt));
                        for (size_t i = 0; i < learn.size(); ++i)
                        {
                            TPicture p(learn[i], false);
                            estimator.BackPropagation(p.AsVector(), (learn[i][0] == 0) ? 1.f : 0.f);
                        }
                    }
                    {
                        TTimer timerLearn("Test Iteration " + ToString(iLearnIt));
                        float error = 0.f;
                        for (size_t i = 0; i < test.size(); ++i)
                        {
                            TPicture p(test[i], false);
                            float result = (test[i][0] == 0) ? 1.f : 0.f;
                            float netResult = estimator.GetOutput(p.AsVector());
                            error += Sqr(result - netResult);
                        }
                        printf("Error %d: %f\n", iLearnIt, error);
                    }
                }
            }
        }
    }
    return 0;
}
