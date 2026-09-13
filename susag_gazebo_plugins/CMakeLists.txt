cmake_minimum_required(VERSION 3.5)
project(susag_gazebo_plugins)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(oscillating_obstacle_plugin SHARED src/oscillating_obstacle_plugin.cpp)
target_link_libraries(oscillating_obstacle_plugin ${GAZEBO_LIBRARIES})

install(
  TARGETS oscillating_obstacle_plugin
  LIBRARY DESTINATION lib
)

ament_package()
