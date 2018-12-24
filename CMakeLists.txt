cmake_minimum_required(VERSION 2.8.3)
project(HMM)

# C++ compile flags
# -std=c++11 -> for c++ standarts
# -Wall for WARNING
# -pedantic -> for warning between c and c++
# -pthread for portability
# -g -O0 -fprofile-arcs -ftest-coverage for code coverage
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
message(STATUS CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE})
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_BUILD_TYPE MATCHES Release)

endif ()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    if (CMAKE_CXX_COMPILER_ID MATCHES GNU)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Woverloaded-virtual -Wwrite-strings -Wno-unused")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -g -O0 -fprofile-arcs -ftest-coverage")
    endif ()
endif ()

# eigen
find_package(Eigen)
if(EIGEN_FOUND)
    message(STATUS "eigen: " ${EIGEN_INCLUDE_DIRS})
    include_directories(include ${EIGEN_INCLUDE_DIRS})

else()
    find_package(Eigen3)
    message(STATUS "eigen2 NOT FOUND ")

    message(STATUS "eigen3: " ${EIGEN3_INCLUDE_DIRS})
    include_directories(include ${EIGEN3_INCLUDE_DIRS})

endif()


# boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS filesystem regex)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    #    add_executable(${PROJECT_NAME} main.cpp)

    #    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif ()


# pthread
# https://stackoverflow.com/questions/5395309/how-do-i-force-cmake-to-include-pthread-option-during-compilation
if (NOT CMAKE_VERSION VERSION_LESS 3.1)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

else ()
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    find_package(Threads REQUIRED)
    if (CMAKE_USE_PTHREADS_INIT)
        set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-pthread")
    endif ()
endif ()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP")
endif ()

# opencv
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
set(CV_LIBS ${OpenCV_LIBS})


add_library(${PROJECT_NAME} src/HMM.cpp)


add_executable(${PROJECT_NAME}_test src/HMM_test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})