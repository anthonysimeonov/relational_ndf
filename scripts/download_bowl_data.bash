#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_test_bowl_data.tar.gz https://www.dropbox.com/s/kdfzeqc1rl60dht/rndf_test_bowl_data.tar.gz?dl=0
# wget -O ndf_bowl_data.tar.gz https://www.dropbox.com/s/q3evi7e39wkhetr/ndf_bowl_data.tar.gz?dl=0
TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data
mkdir -p $TRAIN_DATA_DIR
mv ndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR
tar -xzf ndf_bowl_data.tar.gz
#tar -xzf ndf_occ_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data R-NDF bowls copied to $TRAIN_DATA_DIR"
fi
