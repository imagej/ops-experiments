#!/usr/bin/env bash
# Scripts to build and install native C++ libraries
# Adapted from https://github.com/bytedeco/javacpp-presets
set -eu

if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" YacuDecu
    popd
    exit
fi

case $PLATFORM in
    linux-x86_64)
        #$CMAKE -DCMAKE_BUILD_TYPE=Release \
        #       -DCMAKE_INSTALL_PREFIX="../.." \
        #       -DCMAKE_CXX_COMPILER="/usr/bin/g++" \
        #       -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" ..
		cp ../Makefile Makefile
		mkdir -p ../../lib
		make clean
        make
        make install
        ;;
    macosx-*)
        echo "TODO"
        ;;
    windows-x86_64)
        echo "Windows-x86 64 build"
        cp ../Makefile.windows Makefile.windows
		mkdir -p ../../lib
		nmake -f Makefile.windows clean
        nmake -f Makefile.windows 
        nmake -f Makefile.windows install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac


