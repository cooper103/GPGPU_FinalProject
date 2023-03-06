# GPGPU_FinalProject

Author: Cooper Krauth

Description:
This project contains various GPU and CPU implementations to z-score normalize a machine learning dataset stored as a 2D matrix.

The CPU implementation is an optimized sequential algorithm and is used as a basis against which to compare the GPU results.

Two GPU implemenations were completed using CUDA C, and they outline different approaches to parallelizing the same algorithm. Version 1 implements a thread granularity approach, in which a single thread will compute the mean and standard deviation of an entire column of data. Version 2 utilizes a different approach, using a parallel reduction algorithm. This is done by first transposing the data matrix so that the rows and columns are interchanged. Following this, the partial sums of ecah row are computed and sent back to the CPU, which computes the means and standard deviations. 

Results for different matrix sizes are given in the execution times document, and they show that version 1 achieves up to 15x speedup for the mean computation and 60x speedup for the standard deviation computation.
