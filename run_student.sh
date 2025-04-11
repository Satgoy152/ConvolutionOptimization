#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=runmxnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000m 
#SBATCH --time=10:00
#SBATCH --account=eecs471w25_class
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

module load python/3.10 pytorch/2

prefix=""
options=0

for i in "$@"; do
  case $i in
    --ncu)
      (( options++ ))
      ncu_profile="$(pwd)/profile.ncu-rep"
      [ -n "$SLURM_JOB_ID" ] && ncu_profile="$(pwd)/slurm-$SLURM_JOB_ID.ncu-rep"
      prefix="ncu --page=details --nvtx --nvtx-include Inference/Convolution/ --import-source yes --export $ncu_profile --force-overwrite --set detailed"
      export DISABLE_SANDBOX=1
      shift
      ;;
    --nsys)
      (( options++ ))
      nsys_profile="$(pwd)/profile.nsys-rep"
      [ -n "$SLURM_JOB_ID" ] && nsys_profile="$(pwd)/slurm-$SLURM_JOB_ID.nsys-rep"
      prefix="nsys profile -o $nsys_profile -f true"
      export DISABLE_SANDBOX=1
      shift
      ;;
    --memcheck)
      (( options++ ))
      prefix="compute-sanitizer --kernel-regex kns=_ZN7eecs471"
      export DISABLE_SANDBOX=1
      shift
      ;;
    --racecheck)
      (( options++ ))
      prefix="compute-sanitizer --tool racecheck --kernel-regex kns=_ZN7eecs471"
      export DISABLE_SANDBOX=1
      shift
      ;;
    --synccheck)
      (( options++ ))
      prefix="compute-sanitizer --tool synccheck --kernel-regex kns=_ZN7eecs471"
      export DISABLE_SANDBOX=1
      shift
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

if [ "$options" -gt 1 ]; then
    echo "Multiple running options are not allowed."
    exit 1
fi

$prefix python3 -u support/submission.py "$@"

[ -n "$ncu_profile" ] && [ -f "$ncu_profile" ] && echo "$ncu_profile" >> ~/.local/ncu-proxy
[ -n "$nsys_profile" ] && [ -f "$nsys_profile" ] && echo "$nsys_profile" >> ~/.local/nsys-proxy
