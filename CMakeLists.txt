cmake_minimum_required(VERSION 2.8.3)
project(HMM)

#==== load cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Set variables
include(SetEnv)
include(SetBuildFlag)
include(FindPackageHelper)
include(CmakeUtils)


# 1 set Header path
set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 2 create lib
set(LIB_SRC src/Hmm.cpp)
mc_add_library(HmmTrainer ${LIB_SRC})
mc_install_library(HmmTrainer)

message(STATUS "PROJECT_LIBS: " ${PROJECT_LIBS})

set(LIB_SRC src/Hmm_ros.cpp)
mc_add_library(HmmTrainerRos ${LIB_SRC})
target_link_libraries(HmmTrainerRos HmmTrainer)
mc_install_library(HmmTrainerRos)


message(STATUS "PROJECT_LIBS: " ${PROJECT_LIBS})

set(SRC src/HMM_test.cpp)
mc_add_executable(HmmTest ${SRC})
target_link_libraries(HmmTest HmmTrainer)
mc_install_executable(HmmTest)


# Install targets
include(InstallConfig)
# pack to deb
include(CpackConfig)

