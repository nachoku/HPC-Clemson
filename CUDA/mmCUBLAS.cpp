#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// // CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// // CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        // data[i]=float(i);
        // printf("%f ", data[i]);
        data[i] = rand() / (float)RAND_MAX;
      }
        
    // printf("\n");
}

void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            // printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

__global__
void element(int n, int row_num_b, float *a, float *b, float *c)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n)
  {
    // printf("Thread: %d \n ", i);
    int quot=i/row_num_b;
    int rem=i%row_num_b;
    double temp=0;
    for(int j=0;j<row_num_b;j++)
    {
      // printf("quot:%d - rem:%d \n A: %d: %f \n B: %d: %f \n\n", quot, rem, row_num_b*quot + j, a[row_num_b*quot + j], row_num_b*j + rem, b[row_num_b*j + rem]);
      temp+=a[row_num_b*quot + j] * b[row_num_b*j + rem];
    }
    c[quot*row_num_b + rem]=(float)temp;
    // printf("%f ", c[quot*row_num_b + rem]);
  }
  
  
}

int main(void)
{
  int nIter = 30;
  //Set Matrix Sizes
  sMatrixSize matrix_size;
  matrix_size.uiWA =  160;
  matrix_size.uiHA =  160;
  matrix_size.uiWB =  160;
  matrix_size.uiHB =  160;
  matrix_size.uiWC =  160;
  matrix_size.uiHC =  160;

  printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
  //Number of Elements
  unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
  unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
  unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
  // printf("Size of A: %d \n", size_A);
  // printf("Size of B: %d \n", size_B);
  //Memory Size
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;
  unsigned int mem_size_C = sizeof(float) * size_C;

  //Initialize pointer variables
  float *a, *b, *c, *d_a, *d_b, *d_c;

  //Allocate Space for Matrix on Host- Pointer Variables
  a = (float *)malloc(mem_size_A);
  b = (float *)malloc(mem_size_B);
  c = (float *)malloc(mem_size_C);
  
  //Allocate Space for Matrix on Device - Pointer Variables
  cudaMalloc(&d_a, mem_size_A); 
  cudaMalloc(&d_b, mem_size_B);
  cudaMalloc(&d_c, mem_size_C);
  //Fill Elements
  randomInit(a, size_A);
  randomInit(b, size_B);  

  //Memcopy Host to Device- Operators
  cudaMemcpy(d_a, a, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, mem_size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, mem_size_C, cudaMemcpyHostToDevice);

  unsigned int N=size_C;

  // Perform Matrix Multiplication & Record
  cublasHandle_t handle;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Warmup Execution
  cudaDeviceProp deviceProp;

  int block_size = (deviceProp.major < 2) ? 16 : 32;

  dim3 threads(block_size, block_size);
  dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
  element<<<(N+255)/256, 256>>>(N, matrix_size.uiHB, d_a, d_b, d_c);

  //Actual execution
  cudaEventRecord(start, NULL);
  for (int j = 0; j < nIter; j++)
  {
    element<<<(N+255)/256, 256>>>(N, matrix_size.uiHB, d_a, d_b, d_c);
  }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  
  printf( "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

  //Memcopy Device to Host- Result
  cudaMemcpy(c, d_c, mem_size_C, cudaMemcpyDeviceToHost);


  float *reference = (float *)malloc(mem_size_C);
  matrixMulCPU(reference, a, b, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);

bool resCUBLAS = sdkCompareL2fe(reference, c, size_C, 1.0e-6f);

    if (resCUBLAS != true)
    {
  printDiff(reference, c, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    }

    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

  //Release Resources
  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(d_c);
  
  // free(a);
  // free(b);
  // free(c);
  cudaDeviceReset();
}