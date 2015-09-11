#pragma once

#include <vector>

#include "types.h"
#include "timer.h"
#include "str.h"

typedef std::vector<ui8> TUi8Data;
typedef std::vector<TUi8Data> TRows;

struct TCSVReader
{
    TRows m_rows;
    TFileReader m_fileReader;

    TCSVReader(const std::string& filename, bool verbose, size_t limit = std::numeric_limits<size_t>::max())
        : m_fileReader(filename)
    {
        TTimer timer("CVSReader '" + filename + "' " + std::to_string(limit));
        std::string line;
        if (m_fileReader.ReadLine(&line))
        {
            TStringVector tokens;
			const TUi8Data dummy(tokens.size());
			while (m_fileReader.ReadLine(&line) && (m_rows.size() < limit))
            {
                Split(line, ',', &tokens);
                m_rows.push_back(dummy);
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

    TCSVWriter(const std::string& filename)
        : m_fileWriter(filename)
        , m_first(true)
    {
    }

    void NewLine()
    {
        m_fileWriter.Write("\n");
        m_first = true;
    }

    void Put(const std::string& s)
    {
        if (!m_first)
        {
            m_fileWriter.Write(",");
        }
        m_first = false;
        m_fileWriter.Write(s);
    }
};

