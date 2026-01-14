# PARCO Computing Project (2026-244873)

This folder contains the first deliverable content for the exam of "Introduction to parallel computing" of the acedemic year 2025/2026.

## Repository structure

* **`src/`**: Matrices in .mtx format.
* **`scripts/`**: Contains execution script(bash), the C programs and the .h files containing matrix data in CSR format.
* **`results/`**: (Generated) Output of the execution.
* **`plots/`**: (Generated) Visualizations of the results.

---

#### Follow these instructions to clone the repository and run the simulation.
---
### 1. Clone the repository
Open your terminal and clone the project using Git:

```bash
git clone https://github.com/z4pp4l4/PARCO-Computing-2026-244873
cd PARCO-Computing-2026-244873
```
### 2. Make the script executable

Before running the bash script, ensure that it has the necessary executable permissions. Navigate to the project root (if you aren't there already) and run:
```Bash
chmod +x scripts/final_bash_script.sh
```
to make the script executable.
### 3. Compile the C program implementing parallelization
```Bash
gcc scripts/Final_parallel_code.c -o scripts/Final_parallel_code.exe -fopenmp -lm
```
### Usage:

To execute the code, run the main bash script located in the scripts/ folder. The script compiles the parallel C code and executes it with the specified parameters.

Command Syntax

```Bash
./scripts/final_bash_script.sh <MATRIX_NAME> <SCHED_TYPE> <CHUNK_SIZE> <NUM_THREADS>
```
Parameters description:  
```
MATRIX_NAME	---> The name of the matrix header file (without _csr.h). See "Available Matrices" below.  
SCHED_TYPE	---> OpenMP scheduling type (e.g., static, dynamic, guided).  
CHUNK_SIZE	---> The chunk size for the scheduler (integer).  
NUM_THREADS	---> Number of parallel threads to utilize.  
```
Available Matrices:
Based on the header files in the scripts/ directory, you can use the following names for the <MATRIX_NAME> argument:  
```
    nemeth19  
    nemeth05  
    Trefethen_2000  
    tols2000  
    bcsstk05  
    bcsstm05  
    dataset20mfeatpixel_10NN  
```
#### Examples for execution:
Here are a few example commands to reproduce specific test cases:

Example 1: Static Scheduling Run the Trefethen_2000 matrix with static scheduling, chunk size 10, on 8 threads:
```Bash
./scripts/final_bash_script.sh Trefethen_2000 static 10 8
```
Example 2: Guided Scheduling Run the nemeth19 matrix with guided scheduling, chunk size 50, on 16 threads:
```Bash
./scripts/final_bash_script.sh nemeth19 guided 50 16
```

