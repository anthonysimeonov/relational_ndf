#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_test_bottle_data.tar.gz https://www.dropbox.com/s/7eknveu18zfp51k/rndf_test_bottle_data.tar.gz?dl=0
# wget -O ndf_bottle_data.tar.gz https://www.dropbox.com/s/n90491hu386pg0y/ndf_bottle_data.tar.gz?dl=0
TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data
mkdir -p $TRAIN_DATA_DIR
mv ndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR
tar -xzf ndf_bottle_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data R-NDF copied to $TRAIN_DATA_DIR"
fi
