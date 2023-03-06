#include "ZNorm.h"
#include "GPUErrors.h"

__global__ void ThreadMean(float* g_A, float* g_mean, const int ny, const int nx)
{
	int global_idx = threadIdx.x + (blockIdx.x * blockDim.x); //computing global thread index 
	float mean = 0; //sets the mean initially to 0 
	if (global_idx < nx)//checking if the thread index is less than the total number of columns
	{
		for (int i = 0; i < ny; i++)//iterates through each element in a column and summates the value
		{
			mean += g_A[global_idx + i * nx];
		}
		mean /= ny; //dividing by the total elements to compute the mean 
		g_mean[global_idx] = mean; //copying the mean value from register memory to global memory
	}

}

__global__ void ThreadStdDev(float* g_A, float* g_mean, float* g_stddev, const int ny, const int nx)
{
	//computing global thread index and initializing the standard deviation register value to 0
	int global_idx = threadIdx.x + (blockIdx.x * blockDim.x);
	float stddev = 0; 
	if (global_idx < nx) //ensuring that the index does not exceed the number of columns
	{
		for (int i = 0; i < ny; i++) //iterates through each element in a column
		{
			stddev += pow((g_A[global_idx + i * nx] - g_mean[global_idx]), 2);//summation of partial standard dev calculations
		}
		g_stddev[global_idx] = sqrt((stddev / (ny - 1)));//final computation completed with the final summation value, and this
		//value is stored in global memory
	}
	
}

//Helper function for implementing GPU matrix column mean and sample standard deviation computations with each thread computing a column mean and standard deviation
__host__ void gpuThreadHelper(float* h_A, float* h_mean, float* ref_mean, float* h_stddev, float* ref_stddev, const int ny, const int nx)
{
	//Global memory pointer to the data matrix
	float* d_A{};

	//Memory size of the matrix data in bytes
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	//GPU global memory pointer to the mean vector
	float* d_Mean{};
	//GPU global memory pointer to the sample standard deviation vector
	float* d_Stddev{};

	//code to allocate device memory for the matrix, mean, and stddev data
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_Mean, nx * sizeof(float)));
	HandleCUDAError(cudaMalloc((void**)&d_Stddev, nx * sizeof(float)));

	//copying the matrix from host to device global memory 
	HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice));
	//thread and block values. ceil() function used to ensure enough total threads are created 
	int blocksPerGrid = ceil(nx / 256);
	int threadsPerBlock = 256;

	//event creation code 
	float ElapsedTime{};
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	HandleCUDAError(cudaEventRecord(start, 0));
	//launching the kernel using the parameters specified above
	ThreadMean << <blocksPerGrid, threadsPerBlock >> > (d_A, d_Mean, ny, nx);
	//code to compute and display elapsed time 
	HandleCUDAError(cudaDeviceSynchronize());
	HandleCUDAError(cudaEventRecord(stop, 0));
	HandleCUDAError(cudaEventSynchronize(stop));
	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Mean: " << ElapsedTime << " msecs" << endl;
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));

	//code to create events and start recording 
	HandleCUDAError(cudaEventRecord(start, 0));
	//kernel launch statement 
	ThreadStdDev << <blocksPerGrid, threadsPerBlock >> > (d_A, d_Mean, d_Stddev, ny, nx);
	HandleCUDAError(cudaDeviceSynchronize());
	//code to compute and display elapsed time 
	HandleCUDAError(cudaEventRecord(stop, 0));
	HandleCUDAError(cudaEventSynchronize(stop));
	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Standard Deviation: " << ElapsedTime << " msecs" << endl;
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));

	//copying results back from device to host 
	HandleCUDAError(cudaMemcpy(h_mean, d_Mean, nx * sizeof(float), cudaMemcpyDeviceToHost));
	HandleCUDAError(cudaMemcpy(h_stddev, d_Stddev, nx * sizeof(float), cudaMemcpyDeviceToHost));
	Verification("testing:", h_mean, ref_mean, nx);
	Verification("testing:", ref_stddev,h_stddev, nx);

	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_Mean));
	HandleCUDAError(cudaFree(d_Stddev));
	HandleCUDAError(cudaDeviceReset());
}
