# q4e-e2e: end-to-end acceptance client for q4e-server (replaces a9_e2e.ps1)

set(TARGET q4e-e2e)

add_executable(${TARGET} main.cpp q4e-e2e-v6.cpp)
target_link_libraries(${TARGET} PRIVATE cpp-httplib vendor::nlohmann llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
