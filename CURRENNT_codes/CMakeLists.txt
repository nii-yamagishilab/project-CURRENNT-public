PROJECT(currennt)

CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

set(BOOST_ROOT "/home/smg/wang/TOOL/bin/boost_220")

FIND_PACKAGE (CUDA)
FIND_PACKAGE (Boost 1.48.0 REQUIRED COMPONENTS program_options system filesystem random thread  REQUIRED)


INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIR} )

LINK_DIRECTORIES (${Boost_LIBRARY_DIR} /home/smg/wang/TOOL/bin/netcdf_220/lib)

SET (CMAKE_BUILD_TYPE Release)
SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-security")

FILE (GLOB_RECURSE src_lib     currennt_lib/*.cpp currennt_lib/*.hpp     currennt_lib/*.h     currennt_lib/*.cu     currennt_lib/*.cuh)
FILE (GLOB_RECURSE src_trainer currennt/*.cpp     currennt/*.hpp         currennt/*.h         currennt/*.cu         currennt/*.cuh)

CUDA_ADD_EXECUTABLE (${PROJECT_NAME} ${src_lib} ${src_trainer})

CUDA_ADD_CUBLAS_TO_TARGET (${PROJECT_NAME})
CUDA_ADD_CUFFT_TO_TARGET (${PROJECT_NAME})
TARGET_LINK_LIBRARIES (${PROJECT_NAME} netcdf)
TARGET_LINK_LIBRARIES (${PROJECT_NAME} ${Boost_LIBRARIES})

