#pragma once

#include <cstdio>

#include <string>

#include "exceptions.h"

struct TFileReader
{
    FILE* m_file;

    TFileReader(const std::string& filename)
    {
        m_file = fopen(filename.c_str(), "rb");
        if (!m_file)
        {
            throw TException((std::string("could not open '") + filename + "'").c_str());
        }
    }

    bool ReadLine(std::string* result)
    {
        const static size_t BUFFER_LEN = 16536;
        char buffer[BUFFER_LEN];
        if (fgets(buffer, BUFFER_LEN, m_file))
        {
            *result = buffer;
            while (!result->empty() && (result->back() == '\n' || result->back() == '\r'))
            {
                result->resize(result->size() - 1);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Eof()
    {
        return feof(m_file);
    }

    char ReadChar()
    {
        char result;
        if (1 == fread(&result, 1, 1, m_file))
        {
            return result;
        }
        else
        {
            throw TException("read failed");
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

    TFileWriter(const std::string& filename)
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

void MkDir(const std::string& name);
