#!/bin/bash
#SBATCH --account=
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00

# Launch the META-Farm task runner for this metajob
task.run
