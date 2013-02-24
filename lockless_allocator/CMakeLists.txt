PROJECT(ll_alloc)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -Wcast-qual -Wmissing-format-attribute -Wlogical-op -Wstrict-aliasing -Wsign-compare -Wdeclaration-after-statement -Wnested-externs -Wdisabled-optimization -Winline -Wundef -Wimplicit -Wunused -Wfloat-equal -Winit-self -Wformat=2 -Wswitch -Wsequence-point -Wparentheses -Wimplicit -Wchar-subscripts -Wredundant-decls -Wstrict-prototypes -Wbad-function-cast -Wpointer-arith -Wwrite-strings -Wno-long-long -Wmissing-declarations -Wmissing-prototypes -Wextra -Wall -pedantic -ggdb3 -std=gnu99 -O3 -fPIC -pthread")

ADD_LIBRARY(
    llalloc STATIC
    ll_alloc.c)
