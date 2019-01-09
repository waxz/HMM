
#=== cmake function ======
function(mc_add_lib arg)
    set(PROJECT_LIBS "PROJECT_LIBS" PARENT_SCOPE)

endfunction()

function(mc_add_library arg)
    message("======= mc_add_library ======= ")
    #    message("===INSTALL_CMAKE_DIR== " ${INSTALL_CMAKE_DIR})
    message("-- LIB_NAME: " ${ARGV0})  # 打印第一个参数里的所有内容
    message("-- LIB_SRC: " ${ARGN})  # 打印第一个参数里的所有内容
    add_library(${ARGV0} SHARED ${ARGN})
    target_include_directories(${ARGV0} PUBLIC
            $<BUILD_INTERFACE:${HEADER_PATH}> # for headers when building
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # for config_impl.hpp when building
            $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/${PROJECT_NAME}> # for client in install mode
            $<INSTALL_INTERFACE:${INSTALL_LIB_DIR}> # for config_impl.hpp in install mode
            )
    set(PROJECT_LIBS ${PROJECT_LIBS} "${ARGV0}" PARENT_SCOPE)
endfunction()

function(mc_install_library arg)
    set(LIB_NAME ${ARGV0})

    message("======= mc_export ======= ")
    message("-- LIB_NAME: " ${LIB_NAME} " ${LIB_NAME}Export" " ${LIB_NAME}Targets.cmake")


    install(TARGETS ${LIB_NAME}
            EXPORT ${LIB_NAME}Export
            RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
            LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
            ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
            COMPONENT dev)
    #install(EXPORT ${PROJECT_EXPORT}  DESTINATION ${INSTALL_CMAKE_DIR} NAMESPACE ${PROJECT_NAME}:: )
    # This "exports" all targets which have been put into the export set
    install(EXPORT ${LIB_NAME}Export
            DESTINATION ${INSTALL_CMAKE_DIR}
            FILE ${LIB_NAME}Targets.cmake)
endfunction()

function(mc_add_executable arg)
    message("======= mc_add_executable ======= ")
    #    message("===INSTALL_CMAKE_DIR== " ${INSTALL_CMAKE_DIR})
    message("-- executable name: " ${ARGV0})  # 打印第一个参数里的所有内容
    message("-- executable SRC: " ${ARGN})  # 打印第一个参数里的所有内容
    add_executable(${ARGV0} ${ARGN})

    target_include_directories(${ARGV0} PUBLIC
            $<BUILD_INTERFACE:${HEADER_PATH}> # for headers when building
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # for config_impl.hpp when building
            $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/${PROJECT_NAME}> # for client in install mode
            $<INSTALL_INTERFACE:${INSTALL_LIB_DIR}> # for config_impl.hpp in install mode
            )

endfunction()

function(mc_install_executable arg)
    set(EXE_NAME ${ARGV0})

    message("======= mc_export ======= ")
    message("-- EXE_NAME: " ${EXE_NAME} " ${EXE_NAME}Export" " ${EXE_NAME}Targets.cmake")

    install(TARGETS ${EXE_NAME}
            # In order to export target, uncomment next line
            EXPORT ${EXE_NAME}Export
            RUNTIME DESTINATION "${INSTALL_LIB_DIR}" COMPONENT bin
            LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
            ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
            )

    install(EXPORT ${EXE_NAME}Export
            DESTINATION ${INSTALL_CMAKE_DIR}
            FILE ${EXE_NAME}Targets.cmake)
endfunction()

#======================