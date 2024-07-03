#!/bin/bash

# Note how many times you want job to repeat
NUM_JOBS=3

BASE_JOB_NAME="automatic_training"

# Submit subsequent jobs with dependencies
for i in $(seq 1 $NUM_JOBS); do
    job_name="${BASE_JOB_NAME}_${i}"
    previous_job_name="${BASE_JOB_NAME}_$((i-1))"
    echo "starting job ${i}:"
    qsub -N $job_name -hold_jid $previous_job_name -j y -o "/projectnb/tianlabdl/rsyed/automatic-training/logs/${BASE_JOB_NAME}_${i}.qlog" "job.qsub"
done
