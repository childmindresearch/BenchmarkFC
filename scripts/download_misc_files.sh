#!/bin/bash

set -ex

# HCP subject list previously used in He 2019 paper and others.  But note, we do not use
# this subject list and instead recreate our own according to a similar procedure.
mkdir -p resources/subject_lists 2>/dev/null
wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/v0.16.1-He2019_KRDNN/stable_projects/predict_phenotypes/He2019_KRDNN/replication/input/kr_hcp/He2019_hcp_953_subject_list.txt \
    -O resources/subject_lists/He2019_hcp_953_subject_list.txt

# HCP behavioral column lists used in Li 2019 paper and others.  We use the 58 behavior
# list, but recreate our own column subset lists.
mkdir -p resources/column_lists 2>/dev/null
subsets="58behaviors_age_sex Cognitive_unrestricted Personality_Task_unrestricted Social_Emotion_unrestricted"
for subset in $subsets; do
    wget "https://github.com/ThomasYeoLab/CBIG/raw/refs/tags/v0.9.4-Li2019_GSR/stable_projects/preprocessing/Li2019_GSR/replication/scripts/HCP_lists/${subset}.txt" \
    -O "resources/column_lists/${subset}.txt"
done

# Drop age sex from this file for just 58 behaviors.
head -n 58 resources/column_lists/58behaviors_age_sex.txt > resources/column_lists/58behaviors.txt

# HCP phenotypic data column name lookup table, including full names and long descriptions.
wget "https://wiki.humanconnectome.org/docs/assets/HCP_S1200_DataDictionary_Oct_30_2023.csv" \
    -O "resources/column_lists/HCP_S1200_DataDictionary_Oct_30_2023.csv"
