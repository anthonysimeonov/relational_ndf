#!/bin/bash

set -euo pipefail

if [ -z $RNDF_SOURCE_DIR ]; then echo 'Please source "rndf_env.sh" first'
else
wget -O rndf_relational_demo_demonstrations.tar.gz https://www.dropbox.com/s/5brql2jg0iqvxco/rndf_relational_demo_demonstrations.tar.gz?dl=0
REL_DEMO_DIR=$RNDF_SOURCE_DIR/data/relation_demos/release_demos
mkdir -p $REL_DEMO_DIR
mv rndf_relational_demo_demonstrations.tar.gz $REL_DEMO_DIR
cd $REL_DEMO_DIR
tar -xzf rndf_relational_demo_demonstrations.tar.gz
rm rndf_relational_demo_demonstrations.tar.gz
echo "Robot demonstrations for R-NDF copied to $REL_DEMO_DIR"
fi
