#!/usr/bin/env bash
# Scripts to build and install native C++ libraries
# Adapted from https://github.com/bytedeco/javacpp-presets
set -eu

if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" MKLFFTW
    popd
    exit
fi

case $PLATFORM in
    linux-x86_64)
        $CMAKE -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX="../.." .. 
        make
        make install
        ;;
    macosx-*)
        echo "TODO"
        ;;
    windows-x86_64)
        $CMAKE -G"NMake Makefiles" \
		-DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX="../.." .. 
        nmake
        nmake install
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac


