PROJECT(currennt)

CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

#
# CUDA
#

FIND_PACKAGE (CUDA REQUIRED)
MESSAGE ("-- CUDA_VERSION:        ${CUDA_VERSION}")
MESSAGE ("-- CUDA_INCLUDE_DIRS:   ${CUDA_INCLUDE_DIRS}")
MESSAGE ("-- CUDA_CUDA_LIBRARY:   ${CUDA_CUDA_LIBRARY}")
MESSAGE ("-- CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}")
MESSAGE ("-- CUDA_cublas_LIBRARY: ${CUDA_cublas_LIBRARY}")
MESSAGE ("-- CUDA_CUFFT_LIBRARIES: ${CUDA_CUFFT_LIBRARIES}")
MESSAGE ("-- CUDA_curand_LIBRARY: ${CUDA_curand_LIBRARY}")
SET (CUDA_ALL_LIBRARIES ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY})
INCLUDE_DIRECTORIES ("${CUDA_INCLUDE_DIRS}")
SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

#
# BOOST
#

FIND_PACKAGE (Boost 1.59 REQUIRED COMPONENTS program_options system filesystem random thread)
MESSAGE ("-- Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
MESSAGE ("-- Boost_LIBRARIES:    ${Boost_LIBRARIES}")
INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIRS})

#
# NetCDF
#
MESSAGE ("-- NetCDF Lib:    ${NETCDF_LIB}")
LINK_DIRECTORIES (${NETCDF_LIB})

#
# FLAGS
#

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-security")

#
# PROJECTS
#

FILE (GLOB_RECURSE src_lib     currennt_lib/*.cpp currennt_lib/*.hpp     currennt_lib/*.h     currennt_lib/*.cu     currennt_lib/*.cuh)
FILE (GLOB_RECURSE src_trainer currennt/*.cpp     currennt/*.hpp         currennt/*.h         currennt/*.cu         currennt/*.cuh)
CUDA_ADD_EXECUTABLE (${PROJECT_NAME} ${src_lib} ${src_trainer})
TARGET_LINK_LIBRARIES (${PROJECT_NAME} ${Boost_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_CUFFT_LIBRARIES} netcdf)
