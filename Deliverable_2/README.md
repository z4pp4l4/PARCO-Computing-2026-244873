# PARCO Computing Project – Deliverable 2 (2026-244873)

This folder contains the **second deliverable** for the *Introduction to Parallel Computing* course (a.y. 2025–2026).

The project focuses on **Distributed Sparse Matrix–Vector Multiplication (SpMV)** using **MPI**, with:
- **1D cyclic ROW  partitioning**
- **2D Cartesian BLOCK partitioning**
- **Strong and weak scaling analysis**
- **hybrid MPI + OpenMP analysis results**

The implementation follows **Foster’s Design Methodology** (Partitioning, Communication, Aggregation, Mapping) and evaluates scalability up to **128 MPI processes**.
---

## Repository structure

```bash
Deliverable_2/
│
├── src/
│ └── *.mtx # Sparse matrices in Matrix Market format downloaded
│
├── scripts/
│ ├── MPI_implementation.c # 1D MPI SpMV implementation
│ ├── MPI_impl_2D_partitioning.c # 2D MPI SpMV (Cartesian grid / SUMMA-style)
│ ├── strong_scaling.sh # Strong scaling experiments
│ ├── weak_scaling.sh # Weak scaling experiments
│ ├── *_runtime.sh # Runtime-focused measurements
│ ├── *_static.sh # Static configuration tests
│ ├── *_parallel_for.sh # Hybrid MPI+OpenMP tests
│ ├── generate_weak_mtx.c # Synthetic matrix generator
│ ├── generate_weak_matrices.sh # Weak scaling matrix generation
│ └── download_matrices.sh # Download SuiteSparse matrices
│
├── results/
│ └── # generated via weak/strong scaling results (.txt) 
├── plots/
│ └── contains graphs from the results analysis
└── README.md
```


## Requirements

Since the maximum number of processes that can be used within the code is 128, it is necessary to allocate the resources on the cluster.
```bash
qsub -I -q short_cpuQ -l select=2:ncpus=64:mem=64gb,walltime=01:00:00
```

- MPI implementation (MPICH)
- C compiler with MPI support (`mpicc`)
- Linux-based environment (tested on HPC clusters)

Make sure to have these modules imported, otherwise copy this:
```bash
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load perf
```

---

## How to run:

### 1. Clone the repository

```bash
git clone https://github.com/z4pp4l4/PARCO-Computing-2026-244873
```
Enter the scripts folder:
```bash
cd PARCO-Computing-2026-244873/Deliverable_2/scripts
```
Make the scripts inside the folder executable: 
```bash
chmod +x *.sh
```
Download the .mtx sparse matrices running the following script:
```bash
./download_matrices.sh
```
Real matrices (strong scaling)

Downloaded from the SuiteSparse Matrix Collection, for example:
```bash
# nemeth19.mtx
# Trefethen_20000.mtx
# torso3.mtx
# Ga41As41H72.mtx
```

Generate also synthetic matrices for weak scaling (to see weak scaling)
```bash
./generate_weak_matrices.sh
```
#### Compilation of the C files:
Compile the MPI programs
1D SpMV (cyclic row partitioning)
```bash
mpicc -O3 -std=c99 -fopenmp MPI_implementation.c -o MPI_implementation.out

mpicc -O3 -std=c99 -fopenmp MPI_impl_2D_partitioning.c -o MPI_impl_2D_partitioning.out
```
#### Execution

MPI 1D SpMV

Run the **1D cyclic row-wise MPI implementation**:

```bash
mpirun -np <P> MPI_implementation.out <matrix>
'''
Where; 
    <P> = number of MPI processes; 
    <matrix> = path to the Matrix Market file
'''
#examples:
mpirun -np 8 MPI_implementation.out src/nemeth19.mtx
mpirun -np 32 MPI_implementation.out src/torso3.mtx
```
MPI 2D SpMV

Run the **2D MPI implementation using a Cartesian process grid**:

```bash
mpirun -np <P> MPI_impl_2D_partitioning.out <matrix>
#examples:
mpirun -np 16 MPI_impl_2D_partitioning.out src/Trefethen_20000.mtx
mpirun -np 64 MPI_impl_2D_partitioning.out src/torso3.mtx
```

STRONG SCALING:

Runs the same matrix while increasing the number of MPI processes.
```bash
./strong_scaling.sh
```
Available variants: runtime - static - omp parallel for
```bash
./strong_scaling_runtime.sh 
```
```bash
./strong_scaling_static.sh
```
```bash
./strong_scaling_parallel_for.sh
```
WEAK SCALING

Increases the matrix size proportionally to the number of processes.

```bash
./weak_scaling.sh
```
Available variants: runtime - static - omp parallel for
```bash
./weak_scaling_runtime.sh 
```
```bash
./weak_scaling_static.sh
```
```bash
./weak_scaling_parallel_for.sh
```


### Performance metrics

The following metrics are collected inside the MPI code using MPI_Wtime:
```bash
- Execution time (maximum over all ranks)
- Speedup and parallel efficiency
- GFLOP/s
- Communication vs computation breakdown
- NNZ distribution per rank (min / avg / max)
- Memory footprint per rank
```
Output visulization example:

```bash
********************************************************************************
SUMMARY TABLE:
--------------------------------------------------------------------------------
  P | Time(s) | Comp% | Comm% | GFLOP/s | Speedup |  Eff%  | Imbal | Ghost%
--------------------------------------------------------------------------------
128 |  0.0932 |   0.2 | 102.9 |   0.003 |    0.57 |   0.4 | 1.686 |  75.8
--------------------------------------------------------------------------------
```
