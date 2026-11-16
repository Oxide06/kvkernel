#include<stdio.h>
#include<torch/extension.h>
#include<ATen/cuda/CUDAContext.h>
#include<c10/cuda/CUDAException.h>

// kernel
__global__ void smoke_kernel(const float* x, float* y, int64_t n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n){
    y[idx] = x[idx] + 1.0f;
  }
}

// C++ launcher
torch::Tensor smoke(torch::Tensor x){
  TORCH_CHECK(x.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

  int64_t n = x.numel();

  // allocate output
  torch::Tensor y = torch::empty_like(x);

  const int threads = 256;
  const int blocks = (n + threads - 1)/threads;
  const float* x_ptr = x.data_ptr<float>();
  float* y_ptr = y.data_ptr<float>();

  // current cuda stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // launch
  if(n > 0){
    smoke_kernel<<<blocks, threads, 0, stream>>>(x_ptr, y_ptr, n);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("smoke", &smoke, "smoke kernel");
}