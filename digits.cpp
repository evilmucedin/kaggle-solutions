#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <vector>

#include "gtest/gtest.h"

using namespace std;
using namespace testing;

typedef unsigned char ui8;

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
    throw TException("not implemented");
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

typedef vector<ui8> TUi8Data;
typedef vector<TUi8Data> TRows;

struct TCSVReader
{

    TRows m_rows;
    TFileReader m_fileReader;

    TCSVReader(const string& filename, bool verbose)
        : m_fileReader(filename)
    {
        TTimer timer("CVSReader '" + filename + "'");
        string line;
        if (m_fileReader.ReadLine(&line))
        {
            while (m_fileReader.ReadLine(&line))
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

struct TPicture
{
    const TUi8Data& m_data;
    static const size_t SIZE = 28;

    TPicture(const TUi8Data& data)
        : m_data(data)
    {
    }

    ui8 Get(size_t i, size_t j) const
    {
        return m_data[i*SIZE + j + 1];
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
                printf("%d", GetDigit(i, j));
            printf("\n");
        }
        printf("\n");
    }
};

struct TCommandLineParser
{
    struct TOption
    {
        char m_option;
        string m_longOption;
        string m_description;

        TOption()
        {
        }

        TOption(char option, const string& longOption, const string& description)
            : m_option(option)
            , m_longOption(longOption)
            , m_description(description)
        {
        }
    };
    typedef vector<TOption> TOptions;

    TOptions m_options;
    TStringVector m_args;

    TCommandLineParser(int argc, char* const argv[])
    {
        m_args.resize(argc);
        for (size_t i = 0; i < argc; ++i)
        {
            m_args[i] = argv[i];
        }
    }

    bool Has(char option, const string& longOption, const string& description)
    {
        m_options.push_back( TOption(option, longOption, description) );

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

    bool AutoUsage()
    {
        if (Has('?', "--help", "print usage help"))
        {
            for (size_t i = 0; i < m_options.size(); ++i)
            {
                const TOption& option = m_options[i];
                printf("-%c (--%s) - %s\n", option.m_option, option.m_longOption.c_str(), option.m_description.c_str());
            }
            printf("\n");
            exit(1);
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
            learn->push_back(*toRow);
        else
            test->push_back(*toRow);
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

int main(int argc, char* argv[])
{
    TCommandLineParser parser(argc, argv);
    bool unittests = parser.Has('u', "unittests", "run unittests");
    bool draw = parser.Has('d', "draw", "draw train");
    bool verbose = parser.Has('v', "verbose", "verbose");
    parser.AutoUsage();

    if (draw)
    {
        TCSVReader trainData("train.csv", true);
        for (size_t i = 0; i < trainData.m_rows.size(); ++i)
        {
            TPicture(trainData.m_rows[i]).Draw();
            printf("\n");
        }
    }
    else if (unittests)
    {
        InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    else
    {
        TCSVReader trainData("train.csv", true);
        TRows learn;
        TRows test;
        SplitLearnTest(trainData.m_rows, 0.9, &learn, &test);

        TCosineEstimator estimators[10];
        {
            TTimer timerLearn("Learn");
            for (size_t i = 0; i < learn.size(); ++i)
            {
                estimators[learn[i][0]].Learn( TPicture(learn[i]) );
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
                TPicture p(test[i]);
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
    return 0;
}
