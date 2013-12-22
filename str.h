#pragma once

#include <string>
#include <vector>

#include "types.h"

typedef std::vector<std::string> TStringVector;

void Split(const std::string& line, char sep, TStringVector* result)
{
    result->clear();
    if (!line.empty())
    {
        std::string::const_iterator begin = line.begin();
        std::string::const_iterator now = line.begin();

        while (now < line.end())
        {
            if (*now == sep)
            {
                if (begin != now)
                {
                    result->push_back(std::string(begin, now));
                }
                begin = now + 1;
            }
            ++now;
        }

        if (begin != line.end())
        {
            result->push_back(std::string(begin, line.end()));
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
    EXPECT_EQ(sv[2].length(), 1);
}

bool IsDigit(char ch)
{
    return (ch >= '0') && (ch <= '9');
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
        {
            if (!IsDigit(s[i]))
            {
                throw TException("bad char '" + std::to_string(s[i]) + "'");
            }
            result = 10*result + s[i] - '0';
        }
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
        {
            if (!IsDigit(s[i]))
            {
                throw TException("bad char '" + std::to_string(s[i]) + "'");
            }
            result = 10*result + s[i] - '0';
        }
        return result;
    }
    else
    {
        throw TException("empty string");
    }
}

template<>
unsigned int FromString<unsigned int>(const std::string& s)
{
    unsigned int result = 0;
    if (!s.empty())
    {
        for (size_t i = 0; i < s.length(); ++i)
        {
            if (!IsDigit(s[i]))
            {
                throw TException("bad char '" + std::to_string(s[i]) + "'");
            }
            result = 10*result + s[i] - '0';
        }
        return result;
    }
    else
    {
        throw TException("empty string");
    }
}

template<>
float FromString<float>(const std::string& s)
{
    float result = 0;
    if (1 == sscanf(s.c_str(), "%f", &result))
    {
        return result;
    }
    else
    {
        throw TException("could not cast to float '" + s + "'");
    }
}

