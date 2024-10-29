cmake_minimum_required(VERSION 3.16)

project(Sorry VERSION 0.1 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick)

qt_standard_project_setup()

qt_add_executable(appSorry
  main.cpp
)

qt_policy(
  SET QTP0001 NEW
)

qt_add_qml_module(appSorry
  URI Sorry
  VERSION 1.0
  QML_FILES
    Board.qml
    Card.qml
    Main.qml
    Piece.qml
    PlayerPanel.qml
    Slide.qml
  SOURCES
    sorry_backend.h
    sorry_backend.cpp
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appSorry PROPERTIES
#  MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appSorry
  MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
  MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
  MACOSX_BUNDLE TRUE
  WIN32_EXECUTABLE TRUE
)

target_link_libraries(appSorry
  PRIVATE
  Qt6::Quick
  sorry::agent::mcts
  sorry::agent::rl
  sorry::common
  sorry::engine
)

include(GNUInstallDirs)
install(TARGETS appSorry
  BUNDLE DESTINATION .
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
