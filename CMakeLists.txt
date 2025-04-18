cmake_minimum_required(VERSION 3.16)

project(pink_noise VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/arch:SSE4.2)
else ()
    add_compile_options(-msse4.1)
endif ()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(FFT_DIR rfft)
set(PLOTTER_DIR plotter)
set(PLOTTER_SRC ${PLOTTER_DIR}/plotter.cpp ${PLOTTER_DIR}/plotter.qrc)
set(PRNG_DIR prng)
set(NOISE_GEN_DIR noise_generator)
set(NOISE_GEN_SRC ${NOISE_GEN_DIR}/noisegenerator.cpp ${NOISE_GEN_DIR}/noisegenerator.h)
set(FFT_SRC ${FFT_DIR}/rfft.cpp ${FFT_DIR}/rfft.h ${FFT_DIR}/cpuid.h)
set(PRNG_SRC ${PRNG_DIR}/lfsr.h ${PRNG_DIR}/rng2.h ${PRNG_DIR}/quasi_gauss.h)

INCLUDE_DIRECTORIES(${PLOTTER_DIR} ${NOISE_GEN_DIR} ${FFT_DIR} ${PRNG_DIR})

set(PROJECT_SOURCES
        main.cpp
        widget.cpp
        widget.h
        widget.ui
        ${PLOTTER_SRC}
        ${NOISE_GEN_SRC}
        ${FFT_SRC}
        ${PRNG_SRC}
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(pink_noise
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        plotter/utils.h
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET pink_noise APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(pink_noise SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(pink_noise
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(pink_noise PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.pink_noise)
endif()
set_target_properties(pink_noise PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS pink_noise
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(pink_noise)
endif()
