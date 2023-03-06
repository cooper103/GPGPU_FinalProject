#include "ZNorm.h"

void cpuMean(float* A, float* mean, const int ny, const int nx)
{
	float fSum;
	chrono::time_point<high_resolution_clock> start, end;
	start = high_resolution_clock::now();

	for (int i = 0; i < nx; i++) //loops through each column
	{
		fSum = 0.0f;
		for (int j = 0; j < ny; j++)//loops through each row within each column
		{
			fSum += A[i + j  * nx]; //summation of each element in the row
		}
		mean[i] = fSum /ny;//dividing by the total elements in that row to compute the mean

	}



	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	double computeTime{};
	computeTime = duration_cast<milliseconds>(elasped_seconds).count();
	cout << "CPU Mean Computation time: " << computeTime << " msecs" << endl;
}

void cpuStdDev(float* A, float* mean, float* stddev, const int ny, const int nx)
{
	float fSumSquare;
	chrono::time_point<high_resolution_clock> start, end;
	start = high_resolution_clock::now();
	for (int i = 0; i < nx; i++)//looping through each column
	{
		fSumSquare = 0.0f;//reseting local counter to 0 
		for (int j = 0; j < ny; j++) //iterating through each element (row) in each column 
		{
			fSumSquare += pow((A[i + j * nx] - mean[i]), 2); //partially computing the std dev 
		}
		stddev[i] = sqrt((fSumSquare / (ny - 1)));//finishing the std dev computation and storing in the stddev vector 
	}



	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	double computeTime{};
	computeTime = duration_cast<milliseconds>(elasped_seconds).count();
	cout << "CPU Standard Deviation Computation time: " << computeTime << " msecs" << endl;
}

