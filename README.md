# Continuous Automatic Job Training in SCC
This repository acts as a tutorial for optimizing training models through [Boston University's Shared Computing Cluster (SCC)](https://www.bu.edu/tech/support/research/computing-resources/scc/), the batch system of which is based on the [Sun Grid Engine](https://gridscheduler.sourceforge.net/) (SGE) scheduler. The scripts provided in this repo will allow for automatic job scheduling so that continuous training can occur beyond the maximum time the computing cluster system allows.

## Getting Started
The crucial files in this repository are `job.sh` and `job.qsub`. The roles of both of these files will be explained below. The other important file needed will be the Python script which allows for training. There are some aspects of the script that will need to be changed, as will be explained in the next section

### Job.sh
The job.sh file looks like this:
```
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
```
This file has been designed in a user-friendly manner so that the user can submit a large amount of jobs, each with a set runtime. All jobs submitted will not start until the previous job has been completed. This essentially creates many dependent sub-jobs, effectively simulating one large training job.

The main variables to be altered in this file are `NUM_JOBS` and `HOURS_PER_JOB`. NUM_JOBS is the amount of jobs that will be submitted and HOURS_PER_JOB is simply number of hours allocated to each job. 
As noted in the file, it can be assumed that `Total Training Time = NUM_JOBS * HOURS_PER_JOB`

`LOG_DIRECTORY` should be changed to the path of the directory where stdout/err is expected to be saved.

`BASE_JOB_NAME` should also be changed to whatever the desired name of the job is. Each sub-job/log file will be given a name branching of this base (ex. job_1, job_2 etc.)

Note that running shell scripts frequently causes permission errors in the SCC. The following command should fix any permission-related errors:
`
chmod +x ./job.sh
`

### Job.qsub
The job.qsub file looks like this:
```
#!/bin/bash -l

# Specify the project and resource allocation
#$ -P tianlabdl
#$ -l gpus=1
#$ -l gpu_c=3.5

# Load the required Python module and activate environment
module load python3
source .venv/bin/activate

# Start training
python train.py
```
The qsub file is very standard. Basic specifics for the job are specified at the top (project & resource allocation in this case). After that, any necessary steps to prepare for launching the training script should be completed. This example just loads in the python module and activates the virtual environment. Lastly, the training should start. In this case, the sample training script, `train.py`, is launched.

## Modifying the Training Script
Using batch jobs for training may cause some errors/irritations in the SCC. The following propose solutions to some of these irritations:

### Atomically Saving Checkpoints
If a job is terminated/runs out of time while a checkpoint is being saved, the checkpoint file can be corrupted. This fatal error can be avoided through atomic saving of checkpoints. It is highly recommended to save checkpoints in a similar manner to this code snippet from `train.py`
```
temp_file = f"{checkpoint_directory}/temp.pt"
        save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, temp_file)
        os.replace(temp_file,checkpoint_file)   # Replace old checkpoint only after new one is fully saved
        if os.path.exists(temp_file):     
            os.remove(temp_file)
```
This ensures that should corruption occur, it would happen with the temporary file rather than the actual checkpoint file. A cleanup is also performed to remove temporary file after it replaces the actual checkpoint file.
