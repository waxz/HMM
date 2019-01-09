
# eigen
#  Add the installation prefix of "Eigen" to CMAKE_PREFIX_PATH or set "Eigen_DIR" to a directory containing one of the above files.
find_package(Eigen)
if (EIGEN_FOUND)
    message(STATUS "eigen: " ${EIGEN_INCLUDE_DIRS})
    include_directories(${EIGEN_INCLUDE_DIRS})

else ()
    find_package(Eigen3 REQUIRED)
    message(STATUS "eigen2: NOT FOUND ")

    message(STATUS "eigen3: " ${EIGEN3_INCLUDE_DIRS})
    include_directories(${EIGEN3_INCLUDE_DIRS})

endif ()


# boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS filesystem regex system thread filesystem serialization program_options)
#target_link_libraries(save_test ${Boost_LIBRARIES})
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
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" "${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP")
endif ()

# opencv
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
set(CV_LIBS ${OpenCV_LIBS})
