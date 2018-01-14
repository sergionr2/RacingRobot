project(fast_image_processing)

#**************************************************************************************************
# General cMake settings
#**************************************************************************************************
cmake_minimum_required(VERSION 3.5)

#**************************************************************************************************
# Find Package **************************************************************************************************
find_package(OpenCV 3 REQUIRED)
MESSAGE( STATUS "OpenCV_INCLUDE_DIRS : " ${OpenCV_INCLUDE_DIRS} )
MESSAGE( STATUS "OpenCV_LIB_DIRS : " ${OpenCV_LIB_DIRS} )
MESSAGE( STATUS "OpenCV_LIBS : " ${OpenCV_LIBS} )

find_package(PythonLibs REQUIRED)
MESSAGE( STATUS "PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS} )
MESSAGE( STATUS "PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES} )

find_package(Eigen3 REQUIRED)
# TODO: find_package pybind11

#**************************************************************************************************
# Include **************************************************************************************************
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)

#**************************************************************************************************
# Set variable **************************************************************************************************
SET(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/fast_image_processing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ndarray_converter.cpp
)

#**************************************************************************************************
# Set compiler **************************************************************************************************
SET(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3 -fPIC")
set(PYBIND11_CPP_STANDARD -std=c++14)

#**************************************************************************************************
# Linker **************************************************************************************************
LINK_DIRECTORIES(
  ${OpenCV_LIB_DIR}
)

#**************************************************************************************************
# Make configuration
#**************************************************************************************************
add_library(fast_image_processing SHARED ${SOURCES})
target_link_libraries(fast_image_processing ${PYTHON_LIBRARIES} ${OpenCV_LIBS} ${Eigen3_LIBS})
SET_TARGET_PROPERTIES(fast_image_processing PROPERTIES PREFIX "")

install(TARGETS fast_image_processing DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
