//Tiled Matrix Multiplication
#include "ZNorm.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	int rows = 1 << 14;
	int cols = 1 << 12;
	cout << "Matrix Size: " << rows << "x" << cols << endl;

	//Define Data Matrix
	float* A;
	A = new float[rows * cols];

	//Define a vector to store the mean of the columns of the Data Matrix
	float* MeanVector_cpu;
	MeanVector_cpu = new float[cols];

	//Define a vector to store the standard deviation of the coloumns of the Data Matrix
	float* StdDevVector_cpu;
	StdDevVector_cpu = new float[cols];

	//Define Transpose Matrix
	float* AT;
	AT = new float[rows * cols];

	//GPU Mean Vector
	float* MeanVector_gpu;
	MeanVector_gpu = new float[cols];

	//Define a vector to store the standard deviation of the coloumns of the Data Matrix
	float* StdDevVector_gpu;
	StdDevVector_gpu = new float[cols];

	//Define a vector to store the partial sums returned from the GPU
	float* h_PartialSums;
	h_PartialSums = new float[cols];

	InitializeMatrix(A, rows, cols);
	DisplayMatrix(A, rows, cols);

	//Host Matrix Column Mean Computation
	cpuMean(A, MeanVector_cpu, rows, cols);
	Display("Matrix Column Mean (CPU):", MeanVector_cpu, cols);

	//Host Matrix Column Standard Deviation Computation
	cpuStdDev(A, MeanVector_cpu, StdDevVector_cpu, rows, cols);
	Display("Matrix Column Standard Deviation (CPU: ", StdDevVector_cpu, cols);

	//GPU Matrix Mean Naive Computation
	cout << endl << "Computing Column Means with a Thread per Column " << endl;
	gpuThreadHelper(A, MeanVector_gpu, MeanVector_cpu, StdDevVector_gpu, StdDevVector_cpu, rows, cols);


	//GPU Matrix Mean with Parallel Reduction using the transpose of the input matrix
	cout << endl << "Computing Column Means using GPU Parallel Reduction on each column" << endl;
	gpuPRMeanHelper(A, MeanVector_gpu, MeanVector_cpu, rows, cols, h_PartialSums);

	//code to collect start time
	chrono::time_point<high_resolution_clock> start, end;
	start = high_resolution_clock::now();
	int reduction_blocks = (rows + 256 - 1) / (256); //computes the number of blocks in each grid 

	//cpuMean(h_PartialSums, MeanVector_gpu, reduction_blocks, cols); 
	// //calling the cpu function to add the partial 
	//sums together and store in the mean vector 

	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	double computeTime{};
	computeTime = duration_cast<milliseconds>(elasped_seconds).count();
	cout << "CPU Partial Sums Computation time: " << computeTime << " msecs" << endl;

	delete[] AT;
	delete[] MeanVector_cpu;
	delete[] MeanVector_gpu;
	delete[] StdDevVector_cpu;
	delete[] StdDevVector_gpu;

	return 0;
}