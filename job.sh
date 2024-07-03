#!/bin/bash

NUM_JOBS = 10

timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds

BASE_JOB_NAME="automatic_training_${timestamp}"

# Submit the initial job
job_id=$(qsub -N "${BASE_JOB_NAME}" -j y -o "/projectnb/tianlabdl/rsyed/automatic-training/logs/${BASE_JOB_NAME}.qlog" "job.qsub")

# Submit subsequent jobs with dependencies
for i in $(seq 2 $NUM_JOBS); do
    job_name="${BASE_JOB_NAME}_${i}"
    job_id=$(qsub -N $job_name -hold_jid $job_id -j y -o "/projectnb/tianlabdl/rsyed/automatic-training/logs/${BASE_JOB_NAME}.qlog" "job.qsub")
done
