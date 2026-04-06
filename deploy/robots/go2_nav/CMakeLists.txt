cmake_minimum_required(VERSION 3.12)
project(go2_nav_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

set(UNITREE_DDS_SEARCH_PATHS /usr/local/lib)
if(EXISTS "/usr/local/lib64")
  list(APPEND UNITREE_DDS_SEARCH_PATHS /usr/local/lib64)
endif()

find_library(UNITREE_DDSC_LIB
  NAMES ddsc
  PATHS ${UNITREE_DDS_SEARCH_PATHS}
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(UNITREE_DDSCXX_LIB
  NAMES ddscxx
  PATHS ${UNITREE_DDS_SEARCH_PATHS}
  NO_DEFAULT_PATH
  REQUIRED
)

set(ONNXRUNTIME_LIB_DIR
  ${PROJECT_SOURCE_DIR}/../../../deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib
)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${PROJECT_SOURCE_DIR}/../../../deploy/thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ${UNITREE_DDSC_LIB} ${UNITREE_DDSCXX_LIB} rt pthread
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_nav_ctrl main.cpp)
set_target_properties(go2_nav_ctrl PROPERTIES
  BUILD_RPATH "${UNITREE_DDS_SEARCH_PATHS};${ONNXRUNTIME_LIB_DIR}"
  INSTALL_RPATH "${UNITREE_DDS_SEARCH_PATHS};${ONNXRUNTIME_LIB_DIR}"
)
target_link_options(go2_nav_ctrl PRIVATE "-Wl,--disable-new-dtags")
