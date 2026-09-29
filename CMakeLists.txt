cmake_minimum_required(VERSION 3.16)
project(DLCompiler CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)

# Generate onnx.pb.{h,cc} from onnx.proto (put onnx.proto in third_party/)
protobuf_generate_cpp(ONNX_SRCS ONNX_HDRS third_party/onnx.proto)

add_library(dlc_ir ir/ir.cpp)
add_executable(dlc main.cpp frontend/onnx_frontend.cpp ${ONNX_SRCS})
target_include_directories(dlc PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(dlc dlc_ir ${Protobuf_LIBRARIES})

add_executable(test_ir tests/test_ir.cpp)
target_link_libraries(test_ir dlc_ir)
