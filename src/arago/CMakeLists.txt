# Shared with the lightweight Carlota build. No ONNX or Torch dependency here.
find_package(nlohmann_json 3.11 REQUIRED)
add_library(arago_device SHARED Hardware.cc DecoderLayer.cc Region.cc Binding.cc)
target_link_libraries(arago_device PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(arago_device PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(arago_device PUBLIC cxx_std_17)
