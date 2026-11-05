cmake_minimum_required(VERSION 3.20)
project(BundledImarisWriter LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Compile dependencies into the writer so end users need no native SDK or DLLs.
set(ZLIB_SOURCE "${SOURCES}/zlib")
add_library(ims_zlib STATIC
  ${ZLIB_SOURCE}/adler32.c ${ZLIB_SOURCE}/compress.c ${ZLIB_SOURCE}/crc32.c
  ${ZLIB_SOURCE}/deflate.c ${ZLIB_SOURCE}/gzclose.c ${ZLIB_SOURCE}/gzlib.c
  ${ZLIB_SOURCE}/gzread.c ${ZLIB_SOURCE}/gzwrite.c ${ZLIB_SOURCE}/inflate.c
  ${ZLIB_SOURCE}/infback.c ${ZLIB_SOURCE}/inftrees.c ${ZLIB_SOURCE}/inffast.c
  ${ZLIB_SOURCE}/trees.c ${ZLIB_SOURCE}/uncompr.c ${ZLIB_SOURCE}/zutil.c)
target_include_directories(ims_zlib PUBLIC "${ZLIB_SOURCE}")
if(NOT WIN32)
  target_compile_definitions(ims_zlib PRIVATE HAVE_UNISTD_H)
endif()
set(ZLIB_INCLUDE_DIR "${ZLIB_SOURCE}" CACHE PATH "" FORCE)
set_target_properties(ims_zlib PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/native"
  ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/native")
set(ZLIB_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/native/${CMAKE_STATIC_LIBRARY_PREFIX}ims_zlib${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE FILEPATH "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "" FORCE)
set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16 OFF CACHE BOOL "" FORCE)
add_subdirectory("${SOURCES}/hdf5" hdf5 EXCLUDE_FROM_ALL)
add_dependencies(hdf5-static ims_zlib)

add_library(ims_lz4 STATIC "${SOURCES}/lz4/lib/lz4.c" "${SOURCES}/lz4/lib/lz4hc.c")
target_include_directories(ims_lz4 PUBLIC "${SOURCES}/lz4/lib")
file(GLOB WRITER_SOURCES "${SOURCES}/imaris/c/*.cxx" "${SOURCES}/imaris/writer/*.cxx")
add_library(bpImarisWriter96 SHARED ${WRITER_SOURCES})
if(WIN32)
  set_target_properties(bpImarisWriter96 PROPERTIES PREFIX "")
endif()
target_include_directories(bpImarisWriter96 PRIVATE "${SOURCES}/imaris")
target_compile_definitions(bpImarisWriter96 PRIVATE COMPILE_SHARED_LIBRARY)
target_link_libraries(bpImarisWriter96 PRIVATE hdf5-static ims_lz4 ims_zlib)
if(MSVC)
  target_compile_options(bpImarisWriter96 PRIVATE /bigobj /EHsc)
endif()
if(APPLE)
  set_target_properties(bpImarisWriter96 PROPERTIES INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_NAME_DIR ON)
endif()
