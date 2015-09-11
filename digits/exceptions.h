#pragma once

#include <exception>

#include "types.h"

struct TException : public std::exception
{
    std::string m_message;

    TException()
    {
    }

    TException(const TException& e)
        : m_message(e.m_message)
    {
    }

    TException(const std::string& message)
        : m_message(message)
    {
    }

    virtual const char* what()
    {
        return m_message.c_str();
    }

    ~TException() noexcept
    {
    }
};


