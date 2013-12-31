#pragma once

#include <vector>
#include <string>

#include "types.h"
#include "probEstimator.h"
#include "exceptions.h"

typedef std::vector<float> TFloatVector;

struct TFileWriter;

struct TPicture
{
    static const size_t SIZE = 28;
    static const size_t SIZE2MUL = 4;
    static const size_t SIZE2 = SIZE/SIZE2MUL;
    // static const size_t VECTOR_SIZE = SIZE*SIZE;
    static const size_t VECTOR_SIZE0 = 1 + SIZE*SIZE + 2*SIZE2*SIZE2 + 1 + 10 + 10;
    static const size_t VECTOR_SIZE1 = 10;
    static const size_t VECTOR_SIZE = VECTOR_SIZE0 + VECTOR_SIZE1;

	typedef std::vector<ui8> TUi8Data;
    typedef std::vector<TUi8Data> TUi8Matrix;
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
        m_features[index++] = 1.f;
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

    void CalcFeaturesPhase2(const IProbEstimators& probEstimators)
    {
        for (size_t i = 0; i < probEstimators.size(); ++i)
        {
            m_features[VECTOR_SIZE0 + i] = probEstimators[i]->Estimate(*this);
        }
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

    void Write(TFileWriter& fw) const;

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

    void SaveBMP(const std::string& s) const;
};

typedef std::vector<TPicture> TPictures;