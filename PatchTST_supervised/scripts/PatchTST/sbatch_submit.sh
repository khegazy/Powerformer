#!/bin/bash
SCRIPT_NAME=${1}
SCALE=${2}
JOBNAME=${SCRIPT_NAME}

sbatch <<EOT
#!/bin/bash
#SBATCH -A m636 # Account
#SBATCH -C gpu      # Constraint (type of resource)
#SBATCH -q regular  # Queue
#SBATCH -o slurm_logs/job_%j.out  # send stdout to OUTPUT_FILE
#SBATCH -e slurm_logs/job_%j.out  # send stderr to OUTPUT_FILE
#SBATCH -J $JOBNAME
#SBATCH --mail-type=end           # send email when job ends
#SBATCH -G 4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=23:00:00
#SBATCH --exclusive
#SBATCH --dependency=singleton

module load pytorch/2.0.1
export SLURM_CPU_BIND="cores"
srun sh scripts/PatchTST/${SCRIPT_NAME}.sh ${SCALE}
EOT