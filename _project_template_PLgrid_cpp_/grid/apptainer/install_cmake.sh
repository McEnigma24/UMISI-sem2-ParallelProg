#!/bin/sh
# Ubuntu apt ships CMake 3.28; CMakeLists.txt needs 4.2+ (and C++26).
set -e
CMAKE_VERSION="${CMAKE_VERSION:-4.2.0}"
ARCHIVE="cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${ARCHIVE}"
curl -fsSL "$URL" | tar -xzf - -C /opt
install -d /usr/local/bin
for bin in cmake ctest cpack; do
    ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/${bin}" "/usr/local/bin/${bin}"
done
cmake --version
