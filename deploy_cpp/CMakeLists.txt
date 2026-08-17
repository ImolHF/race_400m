cmake_minimum_required(VERSION 3.16)
project(g1_race_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pass -DUNITREE_SDK2_DIR=/path/to/unitree_sdk2 when this project is copied
# outside the deployment package.
set(UNITREE_SDK2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/unitree_sdk2" CACHE PATH "unitree_sdk2 checkout")
if(NOT EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set UNITREE_SDK2_DIR to the official unitree_sdk2 checkout.")
endif()

add_subdirectory("${UNITREE_SDK2_DIR}" "${CMAKE_BINARY_DIR}/unitree_sdk2" EXCLUDE_FROM_ALL)

add_executable(g1_read_only src/read_only.cpp)
target_link_libraries(g1_read_only PRIVATE unitree_sdk2)

add_executable(g1_joint_map_report src/joint_map_report.cpp)

option(ENABLE_LIBTORCH_SHADOW "Build TorchScript shadow inference; never publishes motor commands." OFF)
if(ENABLE_LIBTORCH_SHADOW)
  find_package(Torch REQUIRED)
  add_executable(g1_shadow_policy src/shadow_policy.cpp)
  target_link_libraries(g1_shadow_policy PRIVATE unitree_sdk2 ${TORCH_LIBRARIES})
  target_compile_options(g1_shadow_policy PRIVATE ${TORCH_CXX_FLAGS})
endif()
