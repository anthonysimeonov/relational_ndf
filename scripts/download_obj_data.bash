#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_obj_assets.tar.gz https://www.dropbox.com/s/jzj24jig5fmyt5b/rndf_obj_assets.tar.gz?dl=0
OBJ_DESC_DIR=$RNDF_SOURCE_DIR/descriptions/objects
mkdir -p $OBJ_DESC_DIR
mv rndf_obj_assets.tar.gz $OBJ_DESC_DIR
cd $OBJ_DESC_DIR
tar -xzf rndf_obj_assets.tar.gz
rm rndf_obj_assets.tar.gz
echo "Object models for R-NDF copied to $OBJ_DESC_DIR"

wget -O rndf_sdf_assets.tar.gz https://www.dropbox.com/s/2obg5hzzzbt4zqj/rndf_sdf_assets.tar.gz?dl=0
SDF_DIR=$OBJ_DESC_DIR/sdf
mkdir -p $SDF_DIR
mv rndf_sdf_assets.tar.gz $SDF_DIR
cd $SDF_DIR
tar -xzf rndf_sdf_assets.tar.gz
rm rndf_sdf_assets.tar.gz
echo "SDF data for R-NDF copied to $SDF_DIR"

fi
