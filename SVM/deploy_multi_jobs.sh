#!/bin/bash
var1="name"
base_num=0
# 40 for vowels

for var2_value in {0..39};
do
    var1_value="exp-svm-semg-$(($var2_value + $base_num))"

    # Use yq to update the YAML file

    yq eval ".metadata.${var1} = \"${var1_value}\"" -i job_svm_semg.yaml
    yq eval '.spec.template.spec.containers[0].args[0] |= sub("-s [0-9]+", "-s '$var2_value'")' -i job_svm_semg.yaml
    kubectl apply -f job_svm_semg.yaml
    
done