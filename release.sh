#!/bin/bash

RELEASE_DIR="./release/"
FIJI_DIR="$HOME/fiji/Fiji.app/"
JARS_DIR="$FIJI_DIR/jars/"

cp ops-experiments-common/target/ops-experiments-common-0.1.0-SNAPSHOT.jar $RELEASE_DIR
cp ops-experiments-cuda/target/ops-experiments-cuda-0.1.0-SNAPSHOT.jar $RELEASE_DIR
cp ops-experiments-imglib2cache/target/ops-experiments-imglib2cache-0.1.0-SNAPSHOT.jar $RELEASE_DIR

cp ops-experiments-imglib2cache/target/dependency/javacpp-1.5.jar $RELEASE_DIR

