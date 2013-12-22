#pragma once


typedef unsigned char ui8;
static_assert(1 == sizeof(ui8), "ui8 bad size");
typedef unsigned short ui16;
static_assert(2 == sizeof(ui16), "ui16 bad size");
typedef unsigned int ui32;
static_assert(4 == sizeof(ui32), "ui32 bad size");

#if defined(_MSC_VER)
    typedef std::make_signed<size_t>::type ssize_t;
#   define noexcept
#endif
