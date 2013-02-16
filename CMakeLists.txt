PROJECT(kaggle)

set(CMAKE_CXX_FLAGS "-std=c++0x")

ADD_SUBDIRECTORY(
    googletest-read-only
)

INCLUDE_DIRECTORIES(
    googletest-read-only/include
)

ADD_EXECUTABLE(
    digits
    digits.cpp
)

TARGET_LINK_LIBRARIES(
    digits
    gtest
)

