#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_test_container_data.tar.gz https://www.dropbox.com/s/nx9me4y4inwhph1/rndf_test_container_data.tar.gz?dl=0
# wget -O rndf_container_data.tar.gz # TODO: final
TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data
mkdir -p $TRAIN_DATA_DIR
mv ndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR
tar -xzf ndf_container_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data R-NDF copied to $TRAIN_DATA_DIR"
fi
