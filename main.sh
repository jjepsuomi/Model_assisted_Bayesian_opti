
script_name="sampling.sh"
job_name="sample_job"
analysis_directory_root_path="./sampling_analysis/"
analysis_code_path="./"
#run_environment_path="export PATH=\"/scratch/project_2005231/Jonnes_analysis_2023/maatu_env/bin:\$PATH\""
mkdir $analysis_directory_root_path
mkdir $analysis_directory_root_path"logs/"
results_path=$analysis_directory_root_path"results/"
mkdir $analysis_directory_root_path"results/"
echo "analysis directories created..."


# Analysis details
sampling_iterations=1 # How many sampling iterations to do per sampling analysis
prior_sample_count=100
sample_count=50
sample_methods="srs:pu:ilcb:ei:sei"
sampling_repeats=1 # How many times to repeat the experiment, in one task.
# Job details
csc_partition="small"
time_limit="1-00:00:00"
mem_per_task="8G"
cores_per_task=1 # How many cores per task
number_of_parallel_jobs=10 # This means that 10 CPU cores will perform separate 10 parallel analyses.



for i in $(seq 1 $number_of_parallel_jobs); do
    # Using a for loop to repeat an action 100 times
    command_str="python "$analysis_code_path"main.py --sample_methods $sample_methods --sampling_repeats $sampling_repeats --prior_sample_count $prior_sample_count --sample_count $sample_count --sampling_iterations $sampling_iterations --job_id $i --results_path $results_path "
    echo $command_str >> $analysis_directory_root_path"commands.txt"
done


echo "Finished writing Python command list, making Slurm script..."
echo "#!/bin/bash
#SBATCH -J $job_name
#SBATCH -o $analysis_directory_root_path"logs/job_out_%A_%a_tmp.txt"
#SBATCH -e $analysis_directory_root_path"logs/job_err_%A_%a_tmp.txt"
#SBATCH -t $time_limit
#SBATCH --mem=$mem_per_task
#SBATCH --array=1-$number_of_parallel_jobs%$number_of_parallel_jobs
#SBATCH -n 1
#SBATCH -p $csc_partition
#SBATCH --nodes=1
#SBATCH --account=project_$csc_project_id
#SBATCH --cpus-per-task=$cores_per_task

# Below command needed so that logs get produced
export PYTHONUNBUFFERED=TRUE

$run_environment_path
# module load geoconda

(( prev_task = SLURM_ARRAY_TASK_ID - 1 ))
(( start_task = 1 * prev_task + 1 ))
(( end_task = start_task + 1 -1 ))
if [[ \$end_task -gt $number_of_parallel_jobs ]]
then
  end_task=($number_of_parallel_jobs)
fi

# set input file to be processed
for rownum in \$(seq \$start_task \$end_task)
do
  commandline=\$(sed -n \"\$rownum\"p "$analysis_directory_root_path"commands.txt)
  echo \"\$commandline\" | bash
done" > $analysis_directory_root_path$script_name

echo "Finished making Slurm script, launching script..."
sbatch $analysis_directory_root_path$modeling_script_name
echo "Script finish, enjoy the ride!"
fi