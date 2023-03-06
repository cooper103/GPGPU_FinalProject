#include "ZNorm.h"
#include "GPUErrors.h"

__global__ void PartialReduction(float* g_Vector, float* g_PartialSum, const int size)
{
	//computing local and global indices 
	int local_index = threadIdx.x;
	int global_index = threadIdx.x + (blockIdx.x * blockDim.x);
	//computing the starting address of each block 
	float* blockAddress = g_Vector + (blockIdx.x * blockDim.x);
	if (global_index >= size)
	{
		return;
	}
	//for loop with the stride increasing by a factor of 2 each time. this reduces divergence 
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		//computing the index using the stride and local thread index
		int index = 2 * stride * local_index;
		if (index < blockDim.x)
		{
			//with each iteration of the loop, a partial sum with an offset of stide is added to the original index location 
			blockAddress[index] += blockAddress[index + stride];
		}
		__syncthreads(); //barrier synchronization. all threads must reach this point before continuing  
	}
	if (local_index == 0) //only one thread per block will do this 
	{
		g_PartialSum[blockIdx.x] = blockAddress[0]; //this partial sum, which is stored in blockAddress[0] is stored as the partial
		//sum of each block 
	}
}

//Kernel code for transposing the data matrix
__global__ void Transpose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	//x and y indices are generated in the 2D block using the local thread index and block idx 
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)//ensuring that both the x and y indcies are less than the total rows and columns 
	{
		//simple code for each thread to compute 1 transpose using g_Matrix and the x and y indices 
		g_MatrixTranspose[iy * nx + ix] = g_Matrix[ix * ny + iy];
	}
	
}


//Helper function for implementing GPU matrix mean computation with each block computing a column mean
__host__ void gpuPRMeanHelper(float* h_A, float* h_mean, float* ref, const int ny, const int nx, float*h_PartialSums)
{
	//GPU global memory pointer to the matrix
	float* d_A{};
	//Global memory pointer to the transpose of the data matrix
	float* d_transpose{};
	//Memory size of the matrix in bytes
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	//global memory pointer to store the partial sums
	float* d_PartialSums{};
	////Host memory pointer to store the partial sums
	//float* h_PartialSums{};

	//code to allocate device memory 
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes)); 
	HandleCUDAError(cudaMalloc((void**)&d_transpose, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_PartialSums, nx * sizeof(float)));

	//computing the total elements in the matrix and 2D dimensions for the transpose kernel launch 
	int size = nx * ny; 
	int transpose_x = 16;
	int transpose_y = 16;
	dim3 transpose_block(transpose_x, transpose_y); 
	dim3 transpose_grid((nx + transpose_block.x - 1) / transpose_block.x, (ny + transpose_block.y - 1) / transpose_block.y); 

	//computing the blocks per grid for each row, which will be handled by a partial sum algorithm
	int reduction_blocks = (ny + 256 -1) / (256);
	//copying the matrix from host to device 
	HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice));
	
	//event code to create events and record times 
	float ElapsedTime{};
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	HandleCUDAError(cudaEventRecord(start, 0));

	//launching the transpose kernel 
	Transpose << <transpose_grid, transpose_block >> > (d_A, d_transpose, ny, nx); 
	//code to collect and display elapsed time 
	HandleCUDAError(cudaDeviceSynchronize()); 
	HandleCUDAError(cudaEventRecord(stop, 0));
	HandleCUDAError(cudaEventSynchronize(stop));
	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Transpose: " << ElapsedTime << " msecs" << endl;
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));

	//creating pointer for the start of each row of the transposed matrix 
	float* transpose_row;
	int j = 0;
	//event code to record the start event 
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	HandleCUDAError(cudaEventRecord(start, 0));
	for (unsigned int i = 0; i < size; i+= ny)//iterates i through the size of the matrix, incrementing by # of rows 
	{
		transpose_row = (d_transpose + i);//computes pointer to the start of a row 
		//kernel launch statement to concurently launch partial reduction kernels 
		PartialReduction << <reduction_blocks, 256 >> > (transpose_row, (d_PartialSums + j*reduction_blocks), ny);
		j++;
	}
	//code to collect and display the time taken to finish all partial reduction kernels 
	HandleCUDAError(cudaStreamSynchronize(0)); 
	HandleCUDAError(cudaDeviceSynchronize()); 
	HandleCUDAError(cudaDeviceSynchronize());
	HandleCUDAError(cudaEventRecord(stop, 0));
	HandleCUDAError(cudaEventSynchronize(stop));
	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Partial Reduction: " << ElapsedTime << " msecs" << endl;
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));

	//copying partial sum result back 
	cudaMemcpy(h_PartialSums, d_PartialSums, nx * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); 
	

	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_PartialSums));
	HandleCUDAError(cudaDeviceReset());
}
