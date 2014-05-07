/*
  convolution between hog features and hog templates
  author: Yu Xiang
  Date: 04/14/2011
*/

extern "C"
{
#include "convolve.h"
#include "matrix.h"
}
#include <cuda_runtime.h>

/* !!! maximum templates size = BLOCK_SIZE * 3 * sbin !!! */
#define BLOCK_SIZE 21

__constant__ float hog_template[2048];
__global__ void convolve2D(CUMATRIX C, CUMATRIX A, CUMATRIX B, int index);

CUMATRIX fconv(CUMATRIX A, CUMATRIX B)
{
  CUMATRIX A_device;
  CUMATRIX B_device;
  CUMATRIX C, C_device;
  cudaError_t error;

  A_device = alloc_device_cumatrix(A);
  B_device = alloc_device_cumatrix(B);

  // allocate hog response cumatrix
  C.dims_num = 2;
  C.dims = (int*)malloc(sizeof(int)*2);
  C.dims[0] = A.dims[0];
  C.dims[1] = A.dims[1];
  C.length = C.dims[0]*C.dims[1];
  C.data = (float*)malloc(sizeof(float)*C.length);
  C_device = alloc_device_cumatrix(C);

  error = cudaMemset(C_device.data, 0, sizeof(float)*C_device.length);
  if (error != cudaSuccess)
  {
    printf("cudaMemset C_device returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  /* setup execution parameters */
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE+2*(B.dims[0]/2));
  dim3 grid((C.dims[1]+BLOCK_SIZE-1) / BLOCK_SIZE, (C.dims[0]+BLOCK_SIZE-1) / BLOCK_SIZE);

  for(int i = 0; i < B.dims[2]; i++)
  {
    // copy to constant memory
    error = cudaMemcpyToSymbol(hog_template, B.data+i*B.dims[0]*B.dims[1], sizeof(float)*B.dims[0]*B.dims[1]);
    if (error != cudaSuccess)
    {
      printf("cudaMemcpyToSymbol returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }

    convolve2D<<< grid, threads >>>(C_device, A_device, B_device, i);
    cudaThreadSynchronize();
  }

  /* copy result from device to host */
  error = cudaMemcpy(C.data, C_device.data, sizeof(float)*C.length, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
    printf("cudaMemcpy C returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  free_device_cumatrix(&A_device);
  free_device_cumatrix(&B_device);
  free_device_cumatrix(&C_device);
  return C;
}

// implementation of the convolution algorithm described in nvidia
// Image convolution with CUDA for nonseperable kernel
__global__ void convolve2D(CUMATRIX C, CUMATRIX A, CUMATRIX B, int index)
{
  __shared__ float data[3*BLOCK_SIZE][3*BLOCK_SIZE];

  // template size
  int nx = B.dims[1];
  int ny = B.dims[0];

  // feature size
  int fx = A.dims[1];
  int fy = A.dims[0];

  // location in A.data of the current thread
  int x = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y*BLOCK_SIZE + threadIdx.y - ny/2;

  // load data
  float val;
  if(index == B.dims[2]-1)
    val = 0;
  else
    val = 0;

  int dx = x - BLOCK_SIZE;
  int dy = y;
  if(dx >= 0 && dx < fx && dy >= 0 && dy < fy)
    data[threadIdx.x][threadIdx.y] = A.data[index*fx*fy+dx*fy+dy];
  else
    data[threadIdx.x][threadIdx.y] = val;

  dx = x;
  dy = y;
  if(dx >= 0 && dx < fx && dy >= 0 && dy < fy)
    data[threadIdx.x+BLOCK_SIZE][threadIdx.y] = A.data[index*fx*fy+dx*fy+dy];
  else
    data[threadIdx.x+BLOCK_SIZE][threadIdx.y] = val;

  dx = x + BLOCK_SIZE;
  dy = y;
  if(dx >= 0 && dx < fx && dy >= 0 && dy < fy)
    data[threadIdx.x+2*BLOCK_SIZE][threadIdx.y] = A.data[index*fx*fy+dx*fy+dy];
  else
    data[threadIdx.x+2*BLOCK_SIZE][threadIdx.y] = val;
  __syncthreads();

  if(x < fx && y < fy && threadIdx.y >= ny/2 && threadIdx.y < ny/2 + BLOCK_SIZE)
  {
    // location in shared memory
    int xx = threadIdx.x + BLOCK_SIZE - nx/2;
    int yy = threadIdx.y - ny/2;
    float sum = 0;
    for(int i = 0; i < nx; i++)
    {
      for(int j = 0; j < ny; j++)
        sum += hog_template[i*ny+j] * data[xx+i][yy+j];
    }
    C.data[x*fy+y] += sum;
  }
}

/*
int main(int argc, char** argv)
{
  FILE *fp;
  CUMATRIX A, A_device;
  CUMATRIX B, B_device;
  CUMATRIX C, C_device;
  cudaError_t error;

  // load hog features
  fp = fopen(argv[1], "r");
  if(fp == NULL)
  {
    printf("can not open file %s\n", argv[1]);
    return 1;
  }
  A = read_cumatrix(fp);
  fclose(fp);
  A_device = alloc_device_cumatrix(A);

  // generate a random hog template
  B.dims_num = 3;
  B.dims = (int*)malloc(sizeof(int)*3);
  B.dims[0] = 16;
  B.dims[1] = 17;
  B.dims[2] = 32;
  B.length = 16*17*32;
  B.data = (float*)malloc(sizeof(float)*B.length);
  for(int i = 0; i < B.length; i++)
    B.data[i] = 1;
  B_device = alloc_device_cumatrix(B);

  // allocate hog response matrix
  C.dims_num = 2;
  C.dims = (int*)malloc(sizeof(int)*2);
  C.dims[0] = A.dims[0];
  C.dims[1] = A.dims[1];
  C.length = C.dims[0]*C.dims[1];
  C.data = (float*)malloc(sizeof(float)*C.length);
  C_device = alloc_device_cumatrix(C);

  error = cudaMemset(C_device.data, 0, sizeof(float)*C_device.length);
  if (error != cudaSuccess)
  {
    printf("cudaMemset C_device returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  // setup execution parameters
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE+2*(B.dims[0]/2));
  dim3 grid((C.dims[1]+BLOCK_SIZE-1) / BLOCK_SIZE, (C.dims[0]+BLOCK_SIZE-1) / BLOCK_SIZE);

  for(int i = 0; i < B.dims[2]; i++)
  {
    // copy to constant memory
    error = cudaMemcpyToSymbol(hog_template, B.data+i*B.dims[0]*B.dims[1], sizeof(float)*B.dims[0]*B.dims[1]);
    if (error != cudaSuccess)
    {
      printf("cudaMemcpyToSymbol returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }

    convolve2D<<< grid, threads >>>(C_device, A_device, B_device, i);
    cudaThreadSynchronize();
  }

  // copy result from device to host
  error = cudaMemcpy(C.data, C_device.data, sizeof(float)*C.length, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
    printf("cudaMemcpy C returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  fp = fopen(argv[2], "w");
  write_cumatrix(&C, fp);
  fclose(fp);

  free_device_cumatrix(&A_device);
  free_device_cumatrix(&B_device);
  free_device_cumatrix(&C_device);
  free_cumatrix(&A);
  free_cumatrix(&B);
  free_cumatrix(&C);
  return 0;
}
*/
