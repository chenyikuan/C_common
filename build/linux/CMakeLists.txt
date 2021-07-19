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

SET(OPT "-O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT}")

# my own work space
SET(C_WS ~/Documents/linux/ws)
SET(CXX_COMMON ${C_WS}/common)

# set my project here
SET(MYLIB CYK_TOOLS)
SET(SRC_LIST 
	# main.cpp
	${CXX_COMMON}/cyk_singleton.cpp
	${CXX_COMMON}/cykTools.cpp
	${CXX_COMMON}/cykFerns.cpp
	${CXX_COMMON}/cykFerns_with_bf.cpp
	${CXX_COMMON}/cyk_file_tools.cpp
	${CXX_COMMON}/cykSDH.cpp
	${CXX_COMMON}/cykSDH_with_ferns.cpp
)

# ======= for packages =======
FIND_PACKAGE(OpenCV REQUIRED core imgproc highgui objdetect)
FIND_PACKAGE(Armadillo REQUIRED)
FIND_PACKAGE(Boost REQUIRED) # generate boost
# FIND_PACKAGE(Boost REQUIRED system filesystem) # generate boost
FIND_LIBRARY(CURSES NAMES curses PATHS /usr/lib)

# ======= for input dirs =======
INCLUDE_DIRECTORIES(
	${CXX_COMMON}
)
LINK_DIRECTORIES(
	${CXX_COMMON}
)

# ======= for output dirs =======
# executable things
# ADD_EXECUTABLE(${MYEXE} ${SRC_LIST})
# ADD_LIBRARY(${MYLIB} SHARED ${SRC_LIST}) # for this shared option, extern lib are required such as opencv armadillo
ADD_LIBRARY(${MYLIB} STATIC ${SRC_LIST})

# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# SET(EXECUTABLE_OUTPUT_PATH .)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../lib)

TARGET_LINK_LIBRARIES(
	${MYLIB}
	${ARMADILLO_LIBRARIES}
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
	# CURSES
)

