#!/bin/bash

# Total Training Time = NUM_JOB * HOURS_PER_JOB
NUM_JOBS=3
HOURS_PER_JOB=12
LOG_DIRECTORY="/projectnb/tianlabdl/rsyed/automatic-training/logs/"

BASE_JOB_NAME="automatic_training"

# Submit subsequent jobs with dependencies
for i in $(seq 1 $NUM_JOBS); do
    job_name="${BASE_JOB_NAME}_${i}"
    previous_job_name="${BASE_JOB_NAME}_$((i-1))"
    qsub -hold_jid $previous_job_name -j y -N $job_name -o "${LOG_DIRECTORY}${BASE_JOB_NAME}_${i}.qlog" -l h_rt=${HOURS_PER_JOB}:00:00 "job.qsub"
done

# Append all log files in the end
qsub -hold_jid "${BASE_JOB_NAME}_${NUM_JOBS}" -j y -N "append_${BASE_JOB_NAME}" -o "${LOG_DIRECTORY}append.qlog" -v mainlog="${LOG_DIRECTORY}${BASE_JOB_NAME}" -v jobs=$NUM_JOBS "append.qsub"