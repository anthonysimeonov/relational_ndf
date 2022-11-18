#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else

# mini dataset, just to get everything up and running
wget -O rndf_test_mug_data.tar.gz https://www.dropbox.com/s/90esj02256lcp0k/rndf_test_mug_data.tar.gz?dl=0
wget -O rndf_test_bottle_data.tar.gz https://www.dropbox.com/s/7eknveu18zfp51k/rndf_test_bottle_data.tar.gz?dl=0
wget -O rndf_test_bowl_data.tar.gz https://www.dropbox.com/s/kdfzeqc1rl60dht/rndf_test_bowl_data.tar.gz?dl=0
wget -O rndf_test_rack_data.tar.gz https://www.dropbox.com/s/owkpgl80mmoy0dr/rndf_test_rack_data.tar.gz?dl=0
wget -O rndf_test_container_data.tar.gz https://www.dropbox.com/s/r5xhbp0d59yjivx/rndf_test_container_data.tar.gz?dl=0

TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data
mkdir -p $TRAIN_DATA_DIR

mv rndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR

tar -xzf rndf_test_mug_data.tar.gz
tar -xzf rndf_test_bottle_data.tar.gz
tar -xzf rndf_test_bowl_data.tar.gz
tar -xzf rndf_test_rack_data.tar.gz
tar -xzf rndf_test_container_data.tar.gz

rm rndf_*_data.tar.gz

echo "Training data R-NDF copied to $TRAIN_DATA_DIR"
fi
