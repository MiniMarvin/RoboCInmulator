cmake_minimum_required(VERSION 2.6)
project(RoboCInmulator C CXX)


set (CMAKE_CXX_STANDARD 11)
set(CPACK_GENERATOR "TGZ")

set(VREP_FLAGS 
	-DNON_MATLAB_PARSING 
	-DMAX_EXT_API_CONNECTIONS=255 -D__linux
)

set(EXTERN_API 
	remoteApi/extApi.c 
	remoteApi/extApiPlatform.c
)

set(CLASS_SRC
	src/libs/robot_agent.cpp
	src/libs/simulation_group.cpp
)

set(CLASS_HEADERS 
	src/libs/robot_agent.hpp
	src/libs/simulation_group.hpp
)

set(LINK_PARTIAL
	include
	remoteApi
	src/libs
)

set(GENETIC_POOL src/libs/genetic_pool.cpp)

############################################################################################
## OPENNN STUFF                                                                            #
############################################################################################
set(CMAKE_CXX_STANDARD 11)
set(CPACK_GENERATOR "TGZ")

## Find boost lib ##########################################################################
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
############################################################################################

message("Using OpenMP")
set(BOOST_FLAGS "-lboost_system -lboost_date_time -lboost_thread -lboost_filesystem")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")#${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ${BOOST_FLAGS}")#${OpenMP_CXX_FLAGS}")
set(CMAEK_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


set(OPENNNDIR /home/minimarvin/Documents/packages/OpenNN) ## The path for OpenNN Install

include_directories(${OPENNNDIR}/opennn)
include_directories(${OPENNNDIR}/tinyxml2)


include_directories(include)
include_directories(remoteApi)
include_directories(src/libs)

add_executable(simulation ${EXTERN_API} src/simulation_1x1.cpp)
add_executable(test_sim ${EXTERN_API} ${CLASS_SRC} src/tests/test_simulation_group.cpp)
add_executable(test_agent ${EXTERN_API} ${CLASS_SRC} src/tests/test_robotagent.cpp)
add_executable(test_genomes src/tests/test_gen.cpp ${GENETIC_POOL})
add_executable(test_load src/tests/test_load.cpp ${GENETIC_POOL})
add_executable(test_gen_agent src/tests/test_genetic_agent.cpp ${GENETIC_POOL} ${EXTERN_API} ${CLASS_SRC})



target_link_libraries(simulation pthread)
target_compile_options(simulation PUBLIC ${VREP_FLAGS})
set_target_properties(simulation PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(test_sim pthread)
target_compile_options(test_sim PUBLIC ${VREP_FLAGS})
set_target_properties(test_sim PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(test_agent pthread)
target_compile_options(test_agent PUBLIC ${VREP_FLAGS})
set_target_properties(test_agent PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(test_gen_agent pthread)
target_compile_options(test_gen_agent PUBLIC ${VREP_FLAGS})
set_target_properties(test_gen_agent PROPERTIES LINKER_LANGUAGE CXX)

# Add the OPENNN Libs for be linked ########################################################
add_library(tinyxml2 ${OPENNNDIR}/tinyxml2/tinyxml2.h ${OPENNNDIR}/tinyxml2/tinyxml2.cpp)

add_library(opennn STATIC IMPORTED)
set_target_properties(opennn PROPERTIES IMPORTED_LOCATION ${OPENNNDIR}/opennn/libopennn.a)

link_directories(${OPENNNDIR}/opennn)
############################################################################################

target_link_libraries(test_genomes opennn tinyxml2)
target_link_libraries(test_load opennn tinyxml2)
target_link_libraries(test_gen_agent opennn tinyxml2)