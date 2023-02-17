cmake_minimum_required(VERSION 3.10)
project(yolov7_plate)
set(OpenCV_DIR /mnt/Gu/softWare/opencv/opencv455/install/lib/cmake/opencv4)
add_definitions(-std=c++11)
add_definitions(-w)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(NCNN_LIBS /mnt/Gu/softWare/NCNN/build/install/lib)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

#NCNN
include_directories(/mnt/Gu/softWare/NCNN/build/install/include/ncnn/)
link_directories(/mnt/Gu/softWare/NCNN/build/install/lib)
# add_executable(yolov7_plate crnn_ncnn.cpp crnn.cpp)
add_executable(yolov7_plate yolov7face.cpp main.cpp crnn.cpp)

# FIND_PACKAGE(Threads REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# target_link_libraries(yolov7_plate nvonnxparser)
target_link_libraries(yolov7_plate ${OpenCV_LIBS})
target_link_libraries(yolov7_plate ncnn)
# target_link_libraries(yolov7_plate ${CMAKE_THREAD_LIBS_INIT})
add_definitions(-O0)
