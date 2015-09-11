#pragma once

#include <string>
#include <vector>

#include "types.h"
#include "exceptions.h"

typedef std::vector<std::string> TStringVector;

void Split(const std::string& line, char sep, TStringVector* result);

inline bool IsDigit(char ch);

template<typename T>
T FromString(const std::string& s)
{
    T::Unimplemented;
}

template<>
ui8 FromString<ui8>(const std::string& s);

template<>
int FromString<int>(const std::string& s);

template<>
unsigned int FromString<unsigned int>(const std::string& s);

template<>
float FromString<float>(const std::string& s);