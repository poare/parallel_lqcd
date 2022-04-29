#!/bin/bash

##ulimit -c unlimited

#SBATCH --job-name=test_julia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=devel
#SBATCH -t 00:30:00
#SBATCH --output=/home/lqcd/poare/parallel_lqcd/test/wombat/slurm_output/%j.out

set -x

bash
module load openmpi/3.1.5

EXE=/home/lqcd/poare/software/julia-1.7.2/bin/julia
#export JULIA_MPI_BINARY="system"
#export JULIA_MPI_PATH=/opt/software/openmpi-3.1.5

home=/home/lqcd/poare/parallel_lqcd/test/wombat
logs=${home}/logs/jltest_${SLURM_JOB_ID}

mkdir ${logs}
#export OMP_NUM_THREADS=6
MPI_RUN=/opt/software/openmpi-3.1.5/bin/mpirun
#MPI_RUN=/home/lqcd/poare/.julia/bin/mpiexecjl

$MPI_RUN -np 4 $EXE ${home}/hello_world.jl > ${logs}/log.txt
