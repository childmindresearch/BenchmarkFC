#!/bin/bash

# Download cifti format fsLR 32k 7 network Schaefer parcellations from Yeo Lab
# repository.

set -ex

resolutions="200 300 400 500 800 1000"
parc_dir="resources/schaefer_parcellations"
base_url="https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti"

rm -r $parc_dir || true 2>/dev/null
mkdir -p $parc_dir

for res in $resolutions; do
    stem="Schaefer2018_${res}Parcels_7Networks_order"
    wget "${base_url}/${stem}.dscalar.nii" -P "${parc_dir}"
    wget "${base_url}/${stem}.dlabel.nii" -P "${parc_dir}"
    wget "${base_url}/${stem}_info.txt" -P "${parc_dir}"
done
