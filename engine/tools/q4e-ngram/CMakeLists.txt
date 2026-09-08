# q4e-ngram: ngram prefetch diag/accept client for q4e-server (replaces ngram_diag.py + ngram_accept.py)

set(TARGET q4e-ngram)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE cpp-httplib vendor::nlohmann ${CMAKE_THREAD_LIBS_INIT} ws2_32)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
