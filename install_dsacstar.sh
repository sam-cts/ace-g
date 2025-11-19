#!/bin/bash

ARCH=$(uname -m)

echo "CPU Architecture: $ARCH"

export OPENCV_INCLUDE_DIR="/usr/include/opencv4"

case "$ARCH" in
    x86_64)
        echo "This is an x86_64 (64-bit Intel/AMD) system."
        export OPENCV_LIBRARY_DIR="/usr/lib/x86_64-linux-gnu"
        ;;
    aarch64)
        echo "This is an AArch64 (64-bit ARM) system."
        export OPENCV_LIBRARY_DIR="/usr/lib/aarch64-linux-gnu"
        ;;
    *)
        echo "Unknown or other architecture: $ARCH"
        echo "exiting"
        exit 1
        ;;
esac

cd dsacstar

pip3 install .