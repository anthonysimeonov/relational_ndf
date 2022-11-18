#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else

wget -O rndf_sdf_assets.tar.gz https://www.dropbox.com/s/2obg5hzzzbt4zqj/rndf_sdf_assets.tar.gz?dl=0
OBJ_DESC_DIR=$RNDF_SOURCE_DIR/descriptions/objects
SDF_DIR=$OBJ_DESC_DIR/sdf
mkdir -p $SDF_DIR
mv rndf_sdf_assets.tar.gz $SDF_DIR
cd $SDF_DIR
tar -xzf rndf_sdf_assets.tar.gz
rm rndf_sdf_assets.tar.gz
echo "SDF data for R-NDF copied to $SDF_DIR"

fi
