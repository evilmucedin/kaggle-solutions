PROJECT(kaggle)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

ADD_SUBDIRECTORY(
    googletest
)

INCLUDE_DIRECTORIES(
    googletest/include
)

ADD_SUBDIRECTORY(
    easybmp
)

ADD_EXECUTABLE(
    digits
    digits.cpp
)

TARGET_LINK_LIBRARIES(
    digits
    gtest
    easybmp
)

IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "Intel" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "i686")
    ADD_SUBDIRECTORY(
        lockless_allocator
    )

    TARGET_LINK_LIBRARIES(
        digits
        llalloc
    )
ENDIF()
