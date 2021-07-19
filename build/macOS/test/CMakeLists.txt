CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT (CYK)
# ADD_SUBDIRECTORY(src)

# compile options
# SET(CMAKE_C_COMPILER g++)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

SET(OPT "-framework Accelerate -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT}")

# my own work space
SET(C_WS ~/Documents/c_ws)
SET(CXX_COMMON ${C_WS}/common)
SET(CXX_COMMON_LIB ${C_WS}/common/build/macOS/lib)

# set my project here
SET(MYEXE run_test)
SET(SRC_LIST 
	main.cpp
)

# ======= for packages =======
FIND_PACKAGE(OpenCV REQUIRED core imgproc highgui objdetect) # generate OpenCV_LIBS
# FIND_PACKAGE(Boost REQUIRED) # generate boost
# FIND_PACKAGE(OpenCV REQUIRED) # generate OpenCV_LIBS
FIND_PACKAGE(Boost REQUIRED system filesystem) # generate boost
# FIND_PACKAGE(Armadillo REQUIRED) # generate boost
# SET(CYK_TOOLS_LIBS libCYK_TOOLS.a) # my own libs
SET(CYK_TOOLS_LIBS libCYK_TOOLS.dylib) # my own libs

# ======= for input dirs =======
INCLUDE_DIRECTORIES(
	${CXX_COMMON}
)
LINK_DIRECTORIES(
	${CXX_COMMON_LIB}
)

# ======= for output dirs =======
# executable things
ADD_EXECUTABLE(${MYEXE} ${SRC_LIST})

# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH .)
# SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

TARGET_LINK_LIBRARIES(
	${MYEXE}
	${CYK_TOOLS_LIBS}
	# ${ARMADILLO_LIBRARIES}
	# ${OpenCV_LIBS}
	${Boost_LIBRARIES}
)

