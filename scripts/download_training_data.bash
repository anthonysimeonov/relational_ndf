#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else

# full dataset
wget -O rndf_mug_data.tar.gz https://www.dropbox.com/s/90esj02256lcp0k/rndf_mug_data.tar.gz?dl=0
wget -O rndf_bottle_data.tar.gz https://www.dropbox.com/s/7eknveu18zfp51k/rndf_bottle_data.tar.gz?dl=0
wget -O rndf_bowl_data.tar.gz https://www.dropbox.com/s/kdfzeqc1rl60dht/rndf_bowl_data.tar.gz?dl=0
wget -O rndf_rack_data.tar.gz https://www.dropbox.com/s/idmv0tsg37hp5dt/syn_rack_pcd_data_easy.tar.gz?dl=0
wget -O rndf_container_data.tar.gz https://www.dropbox.com/s/f01kp6qr0llu885/syn_container_pcd_smaller.tar.gz?dl=0

TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data

mkdir -p $TRAIN_DATA_DIR
mv ndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR

tar -xzf rndf_mug_data.tar.gz
tar -xzf rndf_bottle_data.tar.gz
tar -xzf rndf_bowl_data.tar.gz
tar -xzf rndf_rack_data.tar.gz
tar -xzf rndf_container_data.tar.gz

rm rndf_*_data.tar.gz

echo "Training data R-NDF copied to $TRAIN_DATA_DIR"
fi
