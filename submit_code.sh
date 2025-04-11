#!/bin/bash

if [ "$#" -ne 0 ]; then
        echo "Usage: /scratch/eecs471w25_class_root/eecs471w25_class/$USER/final_project/submit_code.sh"
        exit 1
fi

chmod 700 new_forward.cu
cp -f new_forward.cu /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cu
setfacl -m u:"amrhuss":rwx /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cu
setfacl -m u:"aryanj":rwx /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cu
setfacl -m u:"reetudas":rwx /scratch/eecs471w25_class_root/eecs471w25_class/all_sub/$USER/final_project/$USER.cu