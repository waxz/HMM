# Install headers
# Create 'version.h'
configure_file(${CMAKE_MODULE_PATH}/version.h.in
        "${HEADER_PATH}/version.h" @ONLY)
# version.h + library header

message(STATUS "INSTALL_INCLUDE_DIR: " ${INSTALL_INCLUDE_DIR} "LIBRARY_FOLDER: " ${PROJECT_NAME})
install(DIRECTORY ${HEADER_PATH}/
        DESTINATION "${INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
        FILES_MATCHING PATTERN "*.h"
        )

# Create the <package>Config.cmake.in
configure_file(${CMAKE_SOURCE_DIR}/cmake/Config.cmake.in
        "${PROJECT_CMAKE_FILES}/${PROJECT_NAME}Config.cmake" @ONLY)

# Create the <package>ConfigVersion.cmake.in
configure_file(${CMAKE_SOURCE_DIR}/cmake/ConfigVersion.cmake.in
        "${PROJECT_CMAKE_FILES}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Install <package>Config.cmake and <package>ConfigVersion.cmake files
install(FILES
        "${PROJECT_CMAKE_FILES}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_CMAKE_FILES}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Uninstall targets
configure_file("${CMAKE_SOURCE_DIR}/cmake/Uninstall.cmake.in"
        "${PROJECT_CMAKE_FILES}/Uninstall.cmake"
        IMMEDIATE @ONLY)
add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${PROJECT_CMAKE_FILES}/Uninstall.cmake)
