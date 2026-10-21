set(TARGET q4e-kbench)
add_executable(${TARGET} q4e-kbench.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
