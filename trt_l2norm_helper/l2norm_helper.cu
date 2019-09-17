#include "l2norm_helper.h"


template<typename T>
__global__ void sqrtKernel(const int n, const T* x, T* y)
{
    printf("Unsupported type.");
}

template<>
__global__ void sqrtKernel(
    const int n,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = sqrtf(x[i]);
    }
}

template<>
__global__ void sqrtKernel(
    const int n,
    const __half* x,
    __half* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
         y[i] = hsqrt(x[i]);
    }
}

template<typename T>
__global__ void rsqrtKernel(const int n, const T* x, T* y)
{
    printf("Unsupported type.");
}

template<>
__global__ void rsqrtKernel(
    const int n,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = rsqrtf(x[i]);
    }
}

template<>
__global__ void rsqrtKernel(
    const int n,
    const __half* x,
    __half* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
         y[i] = hrsqrt(x[i]);
    }
}

template<typename T>
__global__ void maxKernel(const int n, const T eps, const T* x, T* y)
{
    printf("Unsupported type.");
}

template<>
__global__ void maxKernel(
    const int n,
    const float eps,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = fmaxf(x[i], eps);
    }
}

template<>
__global__ void maxKernel(
    const int n,
    const __half eps,
    const __half* x,
    __half* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        if (__hgt(x[i], eps))
        {
          y[i] = x[i];
        } else {
          y[i] = eps;
        }
    }
}

template <typename T>
bool executeInference(
    cudaStream_t stream,
    int op_type,
    T eps,
    int batch_size,
    int C,
    int H,
    int W,
    const T* input,
    T* output)
{
    const int length = C * H * W;
    for (int n = 0; n < batch_size; ++n)
    {
        switch(op_type)
        {
          case operation_t::OP_TYPE_MAX:
            maxKernel<<<(length + 511) / 512, 512, 0, stream>>>(length, eps, input, output);
            break;
          case operation_t::OP_TYPE_RSQRT:
            rsqrtKernel<<<(length + 511) / 512, 512, 0, stream>>>(length, input, output);
            break;
          case operation_t::OP_TYPE_SQRT:
            sqrtKernel<<<(length + 511) / 512, 512, 0, stream>>>(length, input, output);
            break;
          default:
            return 1;
        }
        // Move cursors
        input += length;
        output += length;
    }
    return 0;
}

int L2NormHelper::enqueue(
    int batchSize,
    const void* const* inputs,
    void** outputs,
    void* workspace,
    cudaStream_t stream)
{
    switch(mDataType)
    {
      case DataType::kFLOAT:
        if (!executeInference(stream, op_type, eps, batchSize, C, H, W,
                              (const float*)inputs[0], (float*)outputs[0]))
          {
            return 1;
          }
        break;
      case DataType::kHALF:
        if (!executeInference(stream, op_type, (__half)eps, batchSize, C, H, W,
                              (const __half*)inputs[0], (__half*)outputs[0]))
          {
            return 1;
          }
        break;
      default:
        return 1;
    }
    return 0;
}
