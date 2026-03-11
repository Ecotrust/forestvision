#!/bin/bash

# Script to create mosaics from prediction tiles
# Usage: ./create_mosaics.sh <state> --prediction-year <year> [--task <task_name>] [extra_args...]
#   state: State code (e.g., 'or', 'wa')
#   --prediction-year: Required. Year for prediction (e.g., 2024)
#   --task: Optional. Process only the specified task
#   extra_args: Additional arguments passed to create_mosaic.py

state=$1
shift  # Remove state from positional args

# Parse optional arguments
specified_task=""
prediction_year=""
remaining_args=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            specified_task="$2"
            shift 2
            ;;
        --prediction-year)
            prediction_year="$2"
            shift 2
            ;;
        *)
            remaining_args+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$state" ]]; then
    echo "Error: state argument is required"
    exit 1
fi

# Validate state name
if [[ "$state" != "oregon" && "$state" != "washington" ]]; then
    echo "Error: state must be 'oregon' or 'washington' (got: '$state')"
    exit 1
fi

if [[ -z "$prediction_year" ]]; then
    echo "Error: --prediction-year argument is required"
    exit 1
fi

source .env

# Set tasks based on whether a specific task was specified
if [[ -n "$specified_task" ]]; then
    tasks=("$specified_task")
else
    tasks=(
        fortypba
        cancov
        qmd_dom
        ba_ge_3
    )
fi

epsg="EPSG:2992"
if [ "$state" == "washington" ]; then
    epsg="EPSG:2927"
fi

input_dir="data/inference/predictions/${state}/v8/"

for task in "${tasks[@]}"; do
    mosaic_file="data/inference/mosaics/${state}_${task}_mosaic_${prediction_year}.tif"
    if [ -f "$mosaic_file" ]; then
        echo "Mosaic for $task already exists. Skipping..."
        continue
    else
        echo "Creating mosaic for $task..."
        if [ "$task" == "fortypba" ]; then
            python scripts/create_mosaic.py \
                --task "$task" \
                --agg-method mode \
                --crs EPSG:5070 \
                --input-dir "$input_dir" \
                --resampling nearest \
                --out-crs "$epsg" \
                --state "$state" \
                --prediction-year "$prediction_year" \
                "${remaining_args[@]}"
        else
            python scripts/create_mosaic.py \
                --task "$task" \
                --blend-distance 100 \
                --crs EPSG:5070 \
                --input-dir "$input_dir" \
                --resampling bilinear \
                --out-crs "$epsg" \
                --state "$state" \
                --prediction-year "$prediction_year" \
                "${remaining_args[@]}"
        fi
    fi
done
