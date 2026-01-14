"Introduction to parallel programming" repository: 

Contains: 
 - Deliverable 1 code (OpenMP)
 - Deliverable 2 code (MPI + hybrid)


## Objectives

### Deliverable 1: OpenMP-based Sparse Matrix-Vector Multiplication (SpMV)
The objective of Deliverable 1 is to implement a parallel solution for Sparse Matrix-Vector Multiplication (SpMV) using the **OpenMP** parallel programming model. In this deliverable, the main goal is to optimize the computation of SpMV for sparse matrices, utilizing multi-threading techniques to achieve better performance on shared-memory architectures.

#### Key Features:
- **OpenMP parallelization**: Parallelizing the computation of SpMV to exploit multiple CPU cores using OpenMP directives.
- **Optimization of computation**: Enhancing the efficiency of SpMV operations by reducing memory access times and improving data locality.
- **Performance analysis**: Evaluating the performance of the OpenMP implementation using strong scaling, where the problem size remains constant, and the number of threads is increased.

### Deliverable 2: MPI and Hybrid MPI + OpenMP-based Sparse Matrix-Vector Multiplication (SpMV)
Deliverable 2 extends the work done in Deliverable 1 by implementing **MPI** and **Hybrid MPI + OpenMP** solutions for **distributed-memory** environments. The goal is to distribute the computation across multiple nodes (using MPI) and, within each node, parallelize the work using OpenMP. This deliverable includes an in-depth analysis of **scalability**, **communication overhead**, and the comparison between different partitioning strategies.

#### Key Features:
- **MPI-based parallelization**: Implementing a distributed-memory solution using MPI to parallelize the SpMV computation across multiple nodes in a cluster.
- **Hybrid parallelization**: Combining MPI and OpenMP to leverage the strengths of both shared-memory and distributed-memory parallelism.
- **Data partitioning strategies**: Experimenting with 1D and 2D partitioning strategies for sparse matrices to minimize communication costs and maximize computation efficiency.
- **Performance evaluation**: Performing strong and weak scaling experiments to analyze the performance of the hybrid solution under varying problem sizes and process counts.

## Conclusion
This repository demonstrates parallel programming techniques for solving sparse matrix-vector multiplication problems using both **OpenMP** and **MPI**. The hybrid approach combines the benefits of shared-memory and distributed-memory parallelism, making it suitable for large-scale systems.

## References
- **OpenMP Documentation**: [https://www.openmp.org/](https://www.openmp.org/)
- **MPI Documentation**: [https://www.mpi-forum.org/](https://www.mpi-forum.org/)
- **SuiteSparse Matrix Collection**: [https://sparse.tamu.edu/](https://sparse.tamu.edu/)
