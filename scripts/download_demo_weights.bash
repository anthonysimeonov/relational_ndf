#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
WEIGHTS_DIR=$RNDF_SOURCE_DIR/model_weights/ndf_vnn/rndf_weights
mkdir -p $WEIGHTS_DIR
wget -O $WEIGHTS_DIR/ndf_mug.pth https://www.dropbox.com/s/sg2h1un77idf2ns/ndf_mug.pth?dl=0
wget -O $WEIGHTS_DIR/ndf_mug2.pth https://www.dropbox.com/s/ry2ez766g1jgm8a/ndf_mug2.pth?dl=0
wget -O $WEIGHTS_DIR/ndf_bottle.pth https://www.dropbox.com/s/re1y6y2vkjdp1b5/ndf_bottle.pth?dl=0
wget -O $WEIGHTS_DIR/ndf_bowl.pth https://www.dropbox.com/s/riybvdwbdzgjna4/ndf_bowl.pth?dl=0
wget -O $WEIGHTS_DIR/ndf_rack.pth https://www.dropbox.com/s/xn64wnpr15vpmj0/ndf_rack.pth?dl=0
wget -O $WEIGHTS_DIR/ndf_container.pth https://www.dropbox.com/s/5xravyux7h2sfrh/ndf_container.pth?dl=0
fi
