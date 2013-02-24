PROJECT(kaggle)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

ADD_SUBDIRECTORY(
    googletest
)

ADD_SUBDIRECTORY(
    lockless_allocator
)

INCLUDE_DIRECTORIES(
    googletest/include
)

ADD_EXECUTABLE(
    digits
    digits.cpp
)

TARGET_LINK_LIBRARIES(
    digits
    gtest
    llalloc
)
