#include <stdio.h>
#include "timer.h"

__global__ void sq(float* d_in, float* d_out) {
  int idx = threadIdx.x;
  d_out[idx] = d_in[idx] * d_in[idx];
}

int main(int argc, char** argv) {
  const int ARRAY_SIZE = 2048;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; ++i)
    h_in[i] = float(i);
  float h_out[ARRAY_SIZE];

  float* d_in;
  float* d_out;

  GpuTimer timer;
  timer.Start();

  cudaMalloc((void**) &d_in, ARRAY_BYTES);
  cudaMalloc((void**) &d_out, ARRAY_BYTES);

  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  sq<<<1, ARRAY_SIZE>>>(d_in, d_out);

  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // for (int i =0; i < ARRAY_SIZE; i++) {
  //   printf("%f", h_out[i]);
  //   printf(((i % 4) != 3) ? "\t" : "\n");
  // }

  cudaFree(d_in);
  cudaFree(d_out);
  timer.Stop();
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
  return 0;
}
