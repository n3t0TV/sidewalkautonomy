cmake_minimum_required(VERSION 3.0.2)
project(video_processing)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  modules  
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(OpenCV 4.5 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(ZBAR REQUIRED zbar)

#add_message_files(
#  FILES
#)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES video_processing
  CATKIN_DEPENDS message_runtime #roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# common source code
set(COMMON_SRC include/predictor.h src/predictor.cpp
    include/tensor_rt_predictor.h src/tensor_rt_predictor.cpp
    include/opencv_predictor.h src/opencv_predictor.cpp
    include/classification_tracker.h src/classification_tracker.cpp
    include/classifier.h src/classifier.cpp
    include/yolox_decoder.h src/yolox_decoder.cpp
    include/detection_decoder.h src/detection_decoder.cpp
    include/object_tracker.h src/object_tracker.cpp
    include/funct_utils.h src/funct_utils.cpp
    include/ssd_decoder.h src/ssd_decoder.cpp
    include/nlohmann/json.hpp
)

set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser cudart)

# whereiam_node
add_executable(whereiam_node src/whereiam_node.cpp ${COMMON_SRC})
target_link_libraries(whereiam_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${TRT_LIBS})
add_dependencies(whereiam_node ${catkin_EXPORTED_TARGETS})


# vm_analytics_node
add_executable(vm_analytics_node src/vm_analytics_node.cpp
    include/person_analytics.h src/person_analytics.cpp
    include/zbarqrdecoder.h src/zbarqrdecoder.cpp
    include/vm_analytics_utils.h src/vm_analytics_utils.cpp
    ${COMMON_SRC}
    )
target_link_libraries(vm_analytics_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${TRT_LIBS} zbar)
add_dependencies(vm_analytics_node ${catkin_EXPORTED_TARGETS})


# whereiam_node 
install(TARGETS
  whereiam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# vm_analytics_node 
install(TARGETS
  vm_analytics_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# config 
install(DIRECTORY 
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#launch 
install(DIRECTORY 
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
