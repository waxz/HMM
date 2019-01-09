########################################################
###                   PACKAGING                      ###
### This is a quite INCOMPLETE set of variables that ###
### should be set for the various generators.        ###
### Consult the CPack documentations for a full set. ###
########################################################
if (WIN32)
    if (USE_WIX_TOOLSET)
        set(CPACK_GENERATOR "WIX") # this need WiX Tooset installed and a path to candle.exe
    else ()
        set(CPACK_GENERATOR "NSIS") # this needs NSIS installed, and available
    endif ()
    set(CPACK_SOURCE_GENERATOR "ZIP")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CPACK_GENERATOR "PackageMake")
else ()
    set(CPACK_GENERATOR "DEB")
    set(CPACK_SOURCE_GENERATOR "TGZ")
endif ()

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Geoff R. McLane") #required

set(CPACK_PACKAGE_VERSION ${PROJECT_VER})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VER_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VER_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VER_PATCH})


set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

set(CPACK_SOURCE_IGNORE_FILES "${CMAKE_SOURCE_DIR}/build/;${CMAKE_SOURCE_DIR}/.git/")
include(CPack)