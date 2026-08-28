cmake_minimum_required(VERSION 4.0)

project(rea LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rea INTERFACE)

target_include_directories(
    rea
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(rea INTERFACE cxx_std_23)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()

    add_executable(${PROJECT_NAME}_tests tests/test_main.cpp)
    target_link_libraries(${PROJECT_NAME}_tests PRIVATE ${PROJECT_NAME})

    target_compile_options(
        ${PROJECT_NAME}_tests
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wdouble-promotion
            -Wold-style-cast
            -Wcast-align
            -Werror
    )


    add_test(NAME RunAllTests COMMAND ${PROJECT_NAME}_tests)
endif()