#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_test_mug_data.tar.gz https://www.dropbox.com/s/90esj02256lcp0k/rndf_test_mug_data.tar.gz?dl=0
# wget -O ndf_mug_data.tar.gz https://www.dropbox.com/s/42owfein4jtobd5/ndf_mug_data.tar.gz?dl=0
TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data
mkdir -p $TRAIN_DATA_DIR
mv ndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR
tar -xzf ndf_mug_data.tar.gz
#tar -xzf ndf_occ_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data for R-NDF mugs copied to $TRAIN_DATA_DIR"
fi
