#!/bin/bash

set -ex

mkdir -p resources/subject_lists 2>/dev/null
wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/v0.16.1-He2019_KRDNN/stable_projects/predict_phenotypes/He2019_KRDNN/replication/input/kr_hcp/He2019_hcp_953_subject_list.txt \
    -O resources/subject_lists/He2019_hcp_953_subject_list.txt

mkdir -p resources/column_lists 2>/dev/null
subsets="58behaviors_age_sex Cognitive_unrestricted Personality_Task_unrestricted Social_Emotion_unrestricted"
for subset in $subsets; do
    wget "https://github.com/ThomasYeoLab/CBIG/raw/refs/tags/v0.9.4-Li2019_GSR/stable_projects/preprocessing/Li2019_GSR/replication/scripts/HCP_lists/${subset}.txt" \
    -O "resources/column_lists/${subset}.txt"
done

wget "https://wiki.humanconnectome.org/docs/assets/HCP_S1200_DataDictionary_Oct_30_2023.csv" \
    -O "resources/column_lists/HCP_S1200_DataDictionary_Oct_30_2023.csv"
