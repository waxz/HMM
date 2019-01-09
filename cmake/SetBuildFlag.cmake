if (NOT CMAKE_VERSION VERSION_LESS 3.1)
    set(CMAKE_CXX_STANDARD 11)
    #    set(CMAKE_C_STANDARD 99)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    #    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif ()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -pthread -g -O0 -fprofile-arcs -ftest-coverage")
# Build warning with -pedantic https://github.com/ros/rosconsole/issues/9
# example how to set c++ compiler flags for GNU
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

#https://stackoverflow.com/questions/28939652/how-to-detect-sse-sse2-avx-avx2-avx-512-avx-128-fma-kcvi-availability-at-compile
#https://stackoverflow.com/questions/1778538/how-many-gcc-optimization-levels-are-there
if (CMAKE_BUILD_TYPE MATCHES Release)

    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -g -Ofast -ffast-math -O3 -march=native -fopenmp -mavx -mfma")

endif ()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    if (CMAKE_CXX_COMPILER_ID MATCHES GNU)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Woverloaded-virtual -Wwrite-strings -Wno-unused")
        set(CMAKE_CXX_FLAGS_DEBUG "-g3")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -g  -fprofile-arcs -ftest-coverage")
    endif ()
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "-O0")             # 调试包不优化
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ")   # release包优化
message(STATUS "CMAKE_BUILD_TYPE= " ${CMAKE_BUILD_TYPE})

cmake_policy(SET CMP0041 NEW)