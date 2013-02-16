#include <cstdio>

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
    size_t begin = 0;
    for (size_t i = 0; i <= line.length(); ++i)
    {
        if (line[i] == sep || line[i] == 0)
        {
            if (begin != i)
            {
                result->push_back(string(line.c_str() + begin, i - begin));
            }
            begin = i + 1;
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

struct TCSVReader
{
    typedef vector<ui8> TData;

    struct TRow
    {
        ui8 m_label;
        TData m_data;
    };
    typedef vector<TRow> TRows;

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

int main(int argc, char* argv[])
{
    TCommandLineParser parser(argc, argv);
    bool unittests = parser.Has('u', "unittests", "run unittests");
    parser.AutoUsage();

    if (!unittests)
    {
        TCSVReader trainData("train.csv", true);
    }
    else
    {
        InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    return 0;
}
