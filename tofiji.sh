#!/bin/bash

mvn

source release.sh

cp release/* $JARS_DIR
