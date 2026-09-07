# q4e-bench: ON/OFF benchmark client for q4e-server (replaces bench_tps.py --measure/--prefetch)

set(TARGET q4e-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE cpp-httplib vendor::nlohmann ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
