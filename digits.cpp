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

template<typename T>
std::string ToString(const T&)
{
    T::Unimplemented;
}

template<>
std::string ToString<size_t>(const size_t& value)
{
    char buffer[100];
    sprintf(buffer, "%u", (unsigned int)value);
    return buffer;
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

struct TCSVWriter
{
    TFileWriter m_fileWriter;

    TCSVWriter(const string& filename)
        : m_fileWriter(filename)
    {
    }

    void NewLine()
    {
        m_fileWriter.Write("\n");
    }

    void Put(const string& s)
    {
        m_fileWriter.Write(s);
    }
};

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
    const bool unittests = parser.Has('u', "unittests", "run unittests");
    const bool draw = parser.Has('d', "draw", "draw train");
    const bool verbose = parser.Has('v', "verbose", "verbose");
    const bool cosine = parser.Has('c', "cosine", "cosine");
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

    }
    return 0;
}
