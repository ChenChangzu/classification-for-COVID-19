#!/bin/bash
file=$1
save=$2
temp=$3
/opt/fsl/bin/bet $file $temp"/brain_bet" -R -f 0.5 -g 0
/opt/fsl/bin/flirt -in $temp"/brain_bet.nii.gz" -ref /opt/fsl/data/standard/MNI152_T1_2mm_brain -out $temp"/brain_reg" -omat $temp"/mat" -bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
mri_convert $temp"/brain_reg.nii.gz" $save
echo "123456" | sudo -S rm -rf $temp