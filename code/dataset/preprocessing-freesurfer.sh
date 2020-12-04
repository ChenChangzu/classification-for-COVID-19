#!/bin/bash
file=$1
save=$2
temp=$3
recon-all -s sub -i $file -autorecon1
mri_convert $SUBJECTS_DIR"/sub/mri/brainmask.auto.mgz" $save
echo "123456" | sudo -S rm -rf $SUBJECTS_DIR"/sub"