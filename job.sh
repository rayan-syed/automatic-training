#!/bin/bash

# Total Training Time = NUM_JOB * HOURS_PER_JOB
NUM_JOBS=3
HOURS_PER_JOB=1 
LOG_DIRECTORY="/projectnb/tianlabdl/rsyed/automatic-training/logs/"

BASE_JOB_NAME="automatic_training"

# Submit subsequent jobs with dependencies
for i in $(seq 1 $NUM_JOBS); do
    job_name="${BASE_JOB_NAME}_${i}"
    previous_job_name="${BASE_JOB_NAME}_$((i-1))"
    echo "Starting job ${i}:"
    qsub -hold_jid $previous_job_name -j y -N $job_name -o "${LOG_DIRECTORY}${BASE_JOB_NAME}_${i}.qlog" -l h_rt=${HOURS_PER_JOB}:00:00 "job.qsub"
done
