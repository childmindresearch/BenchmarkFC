#!/bin/bash

# Download cifti format fsLR 32k 7 network Schaefer parcellations from Yeo Lab
# repository.

set -ex

resolutions="200 300 400 500 800 1000"
parc_dir="resources/schaefer_parcellations"

rm -r $parc_dir || true 2>/dev/null
mkdir -p $parc_dir

for res in $resolutions; do
    wget "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_${res}Parcels_7Networks_order.dscalar.nii" \
        -P "${parc_dir}"
done
