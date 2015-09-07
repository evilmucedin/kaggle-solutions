#pragma once

#include "str.h"

struct TCommandLineParser
{
    struct TOption
    {
        char m_option;
        std::string m_longOption;
        std::string m_description;
        bool m_isInt;
        int m_intDefault;
        bool m_isString;
        std::string m_stringDefault;

        TOption()
        {
        }

        TOption(char option, const std::string& longOption, const std::string& description, bool isInt, int intDefault, bool isString, const std::string& stringDefault)
            : m_option(option)
            , m_longOption(longOption)
            , m_description(description)
            , m_isInt(isInt)
            , m_intDefault(intDefault)
            , m_isString(isString)
            , m_stringDefault(stringDefault)
        {
        }
    };
    typedef std::vector<TOption> TOptions;

    TOptions m_options;
    TStringVector m_args;
    bool m_error;
    std::string m_strError;

    TCommandLineParser(int argc, char* const argv[])
        : m_error(false)
    {
        m_args.resize(argc);
        for (size_t i = 0; i < argc; ++i)
        {
            m_args[i] = argv[i];
        }
    }

    bool Has(char option, const std::string& longOption, const std::string& description)
    {
        m_options.push_back( TOption(option, longOption, description, false, 0, false, "") );

        static const std::string KEY = "-";
        std::string key = KEY + option;
        static const std::string LONG_KEY = "--";
        std::string longKey = LONG_KEY + longOption;
        for (size_t i = 0; i < m_args.size(); ++i)
        {
            if (m_args[i] == key || m_args[i] == longKey)
            {
                return true;
            }
        }

        return false;
    }

    bool GetInternal(char option, const std::string& longOption, std::string* result)
    {
        static const std::string KEY = "-";
        std::string key = KEY + option;
        static const std::string LONG_KEY = "--";
        std::string longKey = LONG_KEY + longOption;
        for (size_t i = 0; i < m_args.size(); ++i)
        {
            if (m_args[i] == key || m_args[i] == longKey)
            {
                if (i + 1 < m_args.size())
                {
                    *result = m_args[i + 1];
                    return true;
                }
                else
                {
                    m_error = true;
                    m_strError = "not enough arguments";
                }
            }
        }

        return false;
    }

    int GetInt(char option, const std::string& longOption, const std::string& description, int defaultValue)
    {
        m_options.push_back( TOption(option, longOption, description, true, defaultValue, false, "") );

        std::string arg;
        if (GetInternal(option, longOption, &arg))
        {
            try
            {
                return FromString<int>(arg);
            }
            catch (...)
            {
                m_error = true;
                m_strError = "cannot cast to integer '" + arg + "'";
            }
        }

        return defaultValue;
    }

    std::string Get(char option, const std::string& longOption, const std::string& description, const std::string& defaultValue)
    {
        m_options.push_back( TOption(option, longOption, description, false, 0, true, defaultValue) );

        std::string arg;
        if (GetInternal(option, longOption, &arg))
        {
            return arg;
        }

        return defaultValue;
    }

    void AutoUsage()
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
                else if (option.m_isString)
                {
                    printf(" [default='%s']", option.m_stringDefault.c_str());
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

