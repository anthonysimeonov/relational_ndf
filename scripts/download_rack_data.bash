#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_test_rack_data.tar.gz https://www.dropbox.com/s/owkpgl80mmoy0dr/rndf_test_rack_data.tar.gz?dl=0
# wget -O ndf_rack_data.tar.gz  # TODO: final true file
TRAIN_DATA_DIR=$RNDF_SOURCE_DIR/data/training_data
mkdir -p $TRAIN_DATA_DIR
mv ndf_*_data.tar.gz $TRAIN_DATA_DIR
cd $TRAIN_DATA_DIR
tar -xzf ndf_rack_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data R-NDF copied to $TRAIN_DATA_DIR"
fi
