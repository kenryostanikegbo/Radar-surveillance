# ESP-IDF project root for the Radar Surveillance System firmware.
cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS "")

set(PROJECT_NAME radar_surveillance)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(${PROJECT_NAME})

# Embed the dashboard HTML/JS/CSS into the firmware image so the ESP32-S3
# can serve them from the softAP without any external filesystem.
set(DASHBOARD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/main/dashboard")
set(DASHBOARD_FILES
    "${DASHBOARD_DIR}/index.html"
    "${DASHBOARD_DIR}/app.js"
    "${DASHBOARD_DIR}/style.css")
target_add_binary_data(${COMPONENT_LIB} ${DASHBOARD_FILES} TEXT)
