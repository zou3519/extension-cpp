#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>

// Performance is not significantly different, but false saves memory.
// False does not work with unfused pointwise ops.
#define TRAINING (false)

#ifndef PERFOPTS
   #define PERFOPTS (31)
#endif

#define GROUP_GEMM ((PERFOPTS & 1))
#define USE_STREAMS ((PERFOPTS & 2))
#define FUSE_PW ((PERFOPTS & 4))
#define PRE_TRANSPOSE ((PERFOPTS & 8))
#define RECUR_BATCH_SIZE (((PERFOPTS & 16) ? 2 : 1))

namespace {

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));
}

// Pointwise functions
__global__ void pw_biasAdd(float *y, float *bias, int n, int nBias) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] += bias[i % nBias];
}

__global__ void pw_vecAdd(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] * b[i];
}

__global__ void pw_tanh(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = sigmoidf(a[i]);
}

// Unfused LSTM (calling many pointwise kernels).
int LSTM_elementwise_unfused( int hiddenSize,
                               int miniBatch,
                               float * __restrict__ tmp_h,
                               float * __restrict__ tmp_i,
                               float * __restrict__ bias,
                               float * __restrict__ linearGates,
                               float * __restrict__ h_data,
                               float * __restrict__ i_data,
                               float * __restrict__ c_in,
                               float * __restrict__ c_out,
                               bool training,
                               cudaStream_t stream) {
   dim3 blockDim;
   dim3 gridDim;

   int numElements = hiddenSize * miniBatch;

   blockDim.x = 128;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;


   for (int i = 0; i < 4; i++) {
      if (tmp_h != NULL) {
         pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, tmp_i  + i * numElements, tmp_h  + i * numElements, numElements);
         cudaErrCheck(cudaGetLastError());
      }

      pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, bias + i       * hiddenSize, numElements, hiddenSize);
      cudaErrCheck(cudaGetLastError());

      pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, bias + (i + 4) * hiddenSize, numElements, hiddenSize);
      cudaErrCheck(cudaGetLastError());

      if (training) {
         printf("LSTM_elementWise_unfused does not support training\n");
         return 1;
      }
   }

   pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (tmp_i + 0 * numElements, tmp_i + 0 * numElements, numElements);
   cudaErrCheck(cudaGetLastError());

   pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (tmp_i + 1 * numElements, tmp_i + 1 * numElements, numElements);
   cudaErrCheck(cudaGetLastError());

   pw_tanh    <<< gridDim, blockDim, 0, stream >>> (tmp_i + 2 * numElements, tmp_i + 2 * numElements, numElements);
   cudaErrCheck(cudaGetLastError());

   pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (tmp_i + 3 * numElements, tmp_i + 3 * numElements, numElements);
   cudaErrCheck(cudaGetLastError());

   float *in_gate     = tmp_i + 0 * numElements;
   float *forget_gate = tmp_i + 1 * numElements;
   float *in_gate2    = tmp_i + 2 * numElements;
   float *out_gate    = tmp_i + 3 * numElements;

   if (c_in == NULL) {
      pw_vecMul <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, in_gate2, numElements);
      cudaErrCheck(cudaGetLastError());
   }
   else {
      pw_vecMul <<< gridDim, blockDim, 0, stream >>> (forget_gate, forget_gate, c_in, numElements);
      cudaErrCheck(cudaGetLastError());

      pw_vecMul <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, in_gate2, numElements);
      cudaErrCheck(cudaGetLastError());

      pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, forget_gate, numElements);
      cudaErrCheck(cudaGetLastError());
   }

   if (c_out != NULL) {
      cudaErrCheck(cudaMemcpyAsync(c_out, in_gate, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
   }

   pw_tanh <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, numElements);
   cudaErrCheck(cudaGetLastError());

   pw_vecMul <<< gridDim, blockDim, 0, stream >>> (h_data, out_gate, in_gate, numElements);
   cudaErrCheck(cudaGetLastError());

   pw_vecMul <<< gridDim, blockDim, 0, stream >>> (i_data, out_gate, in_gate, numElements);
   cudaErrCheck(cudaGetLastError());

   return 0;
}

// Fused forward kernel
__global__ void elementWise_fp(int hiddenSize, int miniBatch,
                               float *tmp_h,
                               float *tmp_i,
                               float *bias,
                               float *linearGates,
                               float *h_out,
                               float *i_out,
                               float *c_in,
                               float *c_out,
                               bool training) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int numElements = miniBatch * hiddenSize;

   if (index >= numElements) return;

   int batch = index / hiddenSize;
   int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

   float g[4];

   for (int i = 0; i < 4; i++) {
      g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
      // printf("%f, %f\n", tmp_i[i*hiddenSize + gateIndex], tmp_h[i*hiddenSize + gateIndex]);
      g[i] += bias[i * hiddenSize + index % hiddenSize] + bias[(i + 4) * hiddenSize + index % hiddenSize];

      if (training) linearGates[gateIndex + i * hiddenSize] = g[i];
   }


   float in_gate     = sigmoidf(g[0]);
   float forget_gate = sigmoidf(g[1]);
   float in_gate2    = tanhf(g[2]);
   float out_gate    = sigmoidf(g[3]);

   float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);

   c_out[index] = val;

   val = out_gate * tanhf(val);

   h_out[index] = val;
   i_out[index] = val;
}

// Fused forward kernel
__global__ void mi_elementWise_fp(int hiddenSize, int miniBatch,
                               float *tmp_h,
                               float *tmp_i,
                               float *bias,
                               float *alpha,  // 4 * hiddenSize
                               float *beta1,  // 4 * hiddenSize
                               float *beta2,  // 4 * hiddenSize
                               float *linearGates,
                               float *h_out,
                               float *i_out,
                               float *c_in,
                               float *c_out,
                               bool training) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int numElements = miniBatch * hiddenSize;

   if (index >= numElements) return;

   int batch = index / hiddenSize;
   int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

   float g[4];

   float Wx, Uz;
   int biasIdx;
   for (int i = 0; i < 4; i++) {
      Wx = tmp_i[i * hiddenSize + gateIndex];
      Uz = tmp_h[i * hiddenSize + gateIndex];
      biasIdx  = i * hiddenSize + index % hiddenSize;
      g[i] = alpha[biasIdx] * Wx * Uz + beta1[biasIdx] * Wx + beta2[biasIdx] * Uz;
      g[i] += bias[i * hiddenSize + index % hiddenSize] + bias[(i + 4) * hiddenSize + index % hiddenSize];

      if (training) linearGates[gateIndex + i * hiddenSize] = g[i];
   }


   float in_gate     = sigmoidf(g[0]);
   float forget_gate = sigmoidf(g[1]);
   float in_gate2    = tanhf(g[2]);
   float out_gate    = sigmoidf(g[3]);

   float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);

   c_out[index] = val;

   val = out_gate * tanhf(val);

   h_out[index] = val;
   i_out[index] = val;
}

} // namespace

std::vector<at::Tensor> lstm_cuda_forward(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor weights,
    at::Tensor biases,
    int perfopts) {
  if (perfopts == -1) {
    perfopts = 31;
  }
  int group_gemm = perfopts & 1;
  int use_streams = perfopts & 2;
  int fuse_pw = perfopts & 4;
  int pre_transpose = perfopts & 8;
  int recur_batch_size = (perfopts & 16) ? 2 : 1;

  int seqLength = input.size(1);
  int numLayers = hx.size(0);
  int hiddenSize = input.size(-1);
  int miniBatch = input.size(-2);
  bool checkF = true;

   float *h_data;
   float *i_data;
   float *c_data;

   float *T;
   float *T_f;

   float *bias;

   float *tmp_h;
   float *tmp_i;
   float *linearGates;

   cudaStream_t *stream_i;
   cudaStream_t *stream_h;

   cudaEvent_t **events_i;
   cudaEvent_t **events_h;

   // Need a cuBLAS handle.
   cublasHandle_t handle;
   cublasErrCheck(cublasCreate(&handle));

   // Allocate streams/events
   stream_i = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
   stream_h = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));

   // If we don't want to use streams we can launch everything in to the NULL stream
   for (int i = 0; i < numLayers; i++) {
      if (use_streams) {
         cudaErrCheck(cudaStreamCreate(&stream_i[i]));
         // Priority is empirical.
         cudaErrCheck(cudaStreamCreateWithPriority(&stream_h[i], 0, -1));
      }
      else {
         stream_i[i] = NULL;
         stream_h[i] = NULL;
      }
   }


   events_i = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
   events_h = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
   for (int i = 0; i < numLayers; i++) {
      events_i[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
      events_h[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
   }


   // Input/output data
   int numElements = hiddenSize * miniBatch;

   auto i = input;
   auto h = hx;
   auto c = cx;
   auto T_ = weights;
   auto bias_ = biases;

   auto T_f_  = at::zeros({ numLayers * hiddenSize * hiddenSize * 8 }, at::kCUDA);


   // Workspace
   auto tmp_h_ = at::zeros({ 4 * numLayers * numElements }, at::kCUDA);
   auto tmp_i_ = at::zeros({ 4 * seqLength * numElements }, at::kCUDA);

   h_data = h.data<float>();
   i_data = i.data<float>();
   c_data = c.data<float>();
   T = T_.data<float>();
   T_f = T_f_.data<float>();
   bias = bias_.data<float>();
   tmp_h = tmp_h_.data<float>();
   tmp_i = tmp_i_.data<float>();


   // Activations
   if (TRAINING) {
      cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
   }

   // Make sure everything is done before we start the timers
   cudaErrCheck(cudaDeviceSynchronize());

   // Timing starts here
   float elapsedTime;

   cudaEvent_t start, stop;
   cudaErrCheck(cudaEventCreate(&start));
   cudaErrCheck(cudaEventCreate(&stop));

   cudaErrCheck(cudaEventRecord(start));

   float alpha = 1.f;
   float beta  = 0.f;

   const cublasOperation_t transa = (pre_transpose && (seqLength > 1)) ? CUBLAS_OP_N : CUBLAS_OP_T;
   const cublasOperation_t transb = CUBLAS_OP_N;

   // Optimization 4
   if (transa == CUBLAS_OP_N) {
      for (int layer = 0; layer < numLayers; layer++) {
         float *T_i_in = T + layer * hiddenSize * hiddenSize * 8;
         float *T_i_out = T_f + layer * hiddenSize * hiddenSize * 8;

         float *T_h_in = T + layer * hiddenSize * hiddenSize * 8 + hiddenSize * hiddenSize * 4;
         float *T_h_out = T_f + layer * hiddenSize * hiddenSize * 8 + hiddenSize * hiddenSize * 4;

         cublasErrCheck(cublasSetStream(handle, stream_i[layer]));
         cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_i_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_i_out, 4 * hiddenSize));

         cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
         cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_h_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_h_out, 4 * hiddenSize));
      }
   }
   else {
      T_f = T;
   }

   if (transb != CUBLAS_OP_N) {
      printf("Only transb == CUBLAS_OP_N supported\n");
      return {};
   }

   int lStart = 0;
   int lEnd = 0;
   int rStart = 0;
   int rEnd = 0;

   int recurBatchSize = recur_batch_size;

   while (true) {
      // Many layer "scheduling".
      if (lEnd == 0) {
         lStart = 0;
         lEnd = 1;
         rStart = 0;
      }
      else {
         // Move "up" and "left"
         lStart++;
         lEnd++;

         rStart -= recurBatchSize;

         // Over the top or off the left, reset to layer 0
         if (lEnd > numLayers || rStart < 0) {
            rStart += (lStart + 1) * recurBatchSize;

            lStart = 0;
            lEnd = 1;
         }

         // Off the right, step up
         while (rStart >= seqLength && lEnd <= numLayers) {
            lStart++;
            lEnd++;

            rStart -= recurBatchSize;
         }


         // Over the top or off the left, done!
         if (lEnd > numLayers || rStart < 0) {
            break;
         }
      }

      rEnd = rStart + recurBatchSize;
      if (rEnd > seqLength) rEnd = seqLength;

      for (int layer = lStart; layer < lEnd; layer++) {
         cublasErrCheck(cublasSetStream(handle, stream_i[layer]));

         for (int i = rStart; i < rEnd; i++) {
            if (layer > 0) {
               cudaErrCheck(cudaStreamWaitEvent(stream_i[layer], events_h[layer - 1][i], 0));
               cudaErrCheck(cudaEventDestroy(events_h[layer - 1][i]));
            }
         }

         // Optimization 1
         if (group_gemm) {
            cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        4 * hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                        &alpha,
                        &T_f[layer * 8 * hiddenSize * hiddenSize],
                        transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                        i_data + rStart * numElements + layer * seqLength * numElements,
                        hiddenSize,
                        &beta,
                        tmp_i + 4 * rStart * numElements,
                        4 * hiddenSize));
         }
         else {
            for (int igemm =0; igemm < 4; igemm++) {
               cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                        &alpha,
                        &T_f[layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize],
                        transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                        i_data + rStart * numElements + layer * seqLength * numElements,
                        hiddenSize,
                        &beta,
                        tmp_i + 4 * rStart * numElements + igemm * hiddenSize,
                        4 * hiddenSize));
            }
         }

         for (int i = rStart; i < rEnd; i++) {
            cudaErrCheck(cudaEventCreate(&events_i[layer][i], cudaEventDisableTiming));
            cudaErrCheck(cudaEventRecord(events_i[layer][i], stream_i[layer]));
         }

         for (int i = rStart; i < rEnd; i++) {
            cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
            // Optimization 1
            if (group_gemm) {
               cublasErrCheck(cublasSgemm(handle,
                           transa, transb,
                           4 * hiddenSize, miniBatch, hiddenSize,
                           &alpha,
                           &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize],
                           transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                           h_data + i * numElements + layer * (seqLength + 1) * numElements,
                           hiddenSize,
                           &beta,
                           tmp_h + 4 * layer * numElements,
                           4 * hiddenSize));
            }
            else {
               for (int igemm =0; igemm < 4; igemm++) {
                  cublasErrCheck(cublasSgemm(handle,
                              transa, transb,
                              hiddenSize, miniBatch, hiddenSize,
                              &alpha,
                              &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize],
                              transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                              h_data + i * numElements + layer * (seqLength + 1) * numElements,
                              hiddenSize,
                              &beta,
                              tmp_h + 4 * layer * numElements + igemm * hiddenSize,
                              4 * hiddenSize));
               }
            }

            cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_i[layer][i], 0));
            cudaErrCheck(cudaEventDestroy(events_i[layer][i]));

            // Optimization 3
            if (fuse_pw) {
               dim3 blockDim;
               dim3 gridDim;

               blockDim.x = 256;
               gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

               elementWise_fp <<< gridDim, blockDim , 0, stream_h[layer] >>>
                      (hiddenSize, miniBatch,
                       tmp_h + 4 * layer * numElements,
                       tmp_i + 4 * i * numElements,
                       bias + 8 * layer * hiddenSize,
                       TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                       h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       i_data + i * numElements + (layer + 1) * seqLength * numElements,
                       c_data + i * numElements + layer * (seqLength + 1) * numElements,
                       c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       TRAINING);
               cudaErrCheck(cudaGetLastError());
            }
            else {
               LSTM_elementwise_unfused(hiddenSize, miniBatch,
                       tmp_h + 4 * layer * numElements,
                       tmp_i + 4 * i * numElements,
                       bias + 8 * layer * hiddenSize,
                       TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                       h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       i_data + i * numElements + (layer + 1) * seqLength * numElements,
                       c_data + i * numElements + layer * (seqLength + 1) * numElements,
                       c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       TRAINING,
                       stream_h[layer]);
            }
            if (layer != numLayers - 1) {
               cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
               cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer]));
            }
         }
      }
   }
   cudaErrCheck(cudaEventRecord(stop));
   cudaErrCheck(cudaEventSynchronize(stop));
   cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));

   cudaErrCheck(cudaDeviceSynchronize());
   cudaErrCheck(cudaDeviceSynchronize());

   if (TRAINING) cudaErrCheck(cudaFree(linearGates));

   for (int i = 0; i < numLayers; i++) {
      if (stream_i[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_i[i]));
      if (stream_h[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_h[i]));
   }

   free(stream_i);
   free(stream_h);

   for (int i = 0; i < numLayers; i++) {
      free(events_i[i]);
      free(events_h[i]);
   }
   free(events_i);
   free(events_h);

   return { i, h, c, T_, bias_ };
}

std::vector<at::Tensor> milstm_cuda_forward(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor weights,
    at::Tensor biases,
    at::Tensor alphat,
    at::Tensor beta1,
    at::Tensor beta2,
    int perfopts) {
  if (perfopts == -1) {
    perfopts = 31;
  }
  int group_gemm = perfopts & 1;
  int use_streams = perfopts & 2;
  int fuse_pw = perfopts & 4;
  int pre_transpose = perfopts & 8;
  int recur_batch_size = (perfopts & 16) ? 2 : 1;

  int seqLength = input.size(1);
  int numLayers = hx.size(0);
  int hiddenSize = input.size(-1);
  int miniBatch = input.size(-2);
  // int seqLength = 100;
  // int numLayers = 4;
  // int hiddenSize = 512;
  // int miniBatch = 64;
  bool checkF = true;

   float *h_data;
   float *i_data;
   float *c_data;

   float *T;
   float *T_f;

   float *bias;

   float *tmp_h;
   float *tmp_i;
   float *linearGates;

   cudaStream_t *stream_i;
   cudaStream_t *stream_h;

   cudaEvent_t **events_i;
   cudaEvent_t **events_h;

   // Need a cuBLAS handle.
   cublasHandle_t handle;
   cublasErrCheck(cublasCreate(&handle));

   // Allocate streams/events
   stream_i = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
   stream_h = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));

   // If we don't want to use streams we can launch everything in to the NULL stream
   for (int i = 0; i < numLayers; i++) {
      if (use_streams) {
         cudaErrCheck(cudaStreamCreate(&stream_i[i]));
         // Priority is empirical.
         cudaErrCheck(cudaStreamCreateWithPriority(&stream_h[i], 0, -1));
      }
      else {
         stream_i[i] = NULL;
         stream_h[i] = NULL;
      }
   }


   events_i = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
   events_h = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
   for (int i = 0; i < numLayers; i++) {
      events_i[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
      events_h[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
   }


   // Input/output data
   int numElements = hiddenSize * miniBatch;

   // auto i     = at::zeros({ (seqLength) * (numLayers + 1) * numElements }, at::kCUDA);
   // auto h     = at::zeros({ (seqLength + 1) * (numLayers) * numElements }, at::kCUDA);
   // auto c     = at::zeros({ (seqLength + 1) * (numLayers) * numElements }, at::kCUDA);
   // auto bias_ = at::zeros({ numLayers * hiddenSize * 8 }, at::kCUDA);
   // auto T_    = at::zeros({ numLayers * hiddenSize * hiddenSize * 8 }, at::kCUDA);
   auto i = input;
   auto h = hx;
   auto c = cx;
   auto T_ = weights;
   auto bias_ = biases;

   auto T_f_  = at::zeros({ numLayers * hiddenSize * hiddenSize * 8 }, at::kCUDA);


   // Workspace
   auto tmp_h_ = at::zeros({ 4 * numLayers * numElements }, at::kCUDA);
   auto tmp_i_ = at::zeros({ 4 * seqLength * numElements }, at::kCUDA);
   
   float* alpha_data = alphat.data<float>();
   float* beta1_data = beta1.data<float>();
   float* beta2_data = beta2.data<float>();

   h_data = h.data<float>();
   i_data = i.data<float>();
   c_data = c.data<float>();
   T = T_.data<float>();
   T_f = T_f_.data<float>();
   bias = bias_.data<float>();
   tmp_h = tmp_h_.data<float>();
   tmp_i = tmp_i_.data<float>();


   // Activations
   if (TRAINING) {
      cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
   }

   // Make sure everything is done before we start the timers
   cudaErrCheck(cudaDeviceSynchronize());

   // Timing starts here
   float elapsedTime;

   cudaEvent_t start, stop;
   cudaErrCheck(cudaEventCreate(&start));
   cudaErrCheck(cudaEventCreate(&stop));

   cudaErrCheck(cudaEventRecord(start));

   float alpha = 1.f;
   float beta  = 0.f;

   const cublasOperation_t transa = (pre_transpose && (seqLength > 1)) ? CUBLAS_OP_N : CUBLAS_OP_T;
   const cublasOperation_t transb = CUBLAS_OP_N;

   // Optimization 4
   if (transa == CUBLAS_OP_N) {
      for (int layer = 0; layer < numLayers; layer++) {
         float *T_i_in = T + layer * hiddenSize * hiddenSize * 8;
         float *T_i_out = T_f + layer * hiddenSize * hiddenSize * 8;

         float *T_h_in = T + layer * hiddenSize * hiddenSize * 8 + hiddenSize * hiddenSize * 4;
         float *T_h_out = T_f + layer * hiddenSize * hiddenSize * 8 + hiddenSize * hiddenSize * 4;

         cublasErrCheck(cublasSetStream(handle, stream_i[layer]));
         cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_i_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_i_out, 4 * hiddenSize));

         cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
         cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_h_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_h_out, 4 * hiddenSize));
      }
   }
   else {
      T_f = T;
   }

   if (transb != CUBLAS_OP_N) {
      printf("Only transb == CUBLAS_OP_N supported\n");
      return {};
   }

   int lStart = 0;
   int lEnd = 0;
   int rStart = 0;
   int rEnd = 0;

   int recurBatchSize = recur_batch_size;

   while (true) {
      // Many layer "scheduling".
      if (lEnd == 0) {
         lStart = 0;
         lEnd = 1;
         rStart = 0;
      }
      else {
         // Move "up" and "left"
         lStart++;
         lEnd++;

         rStart -= recurBatchSize;

         // Over the top or off the left, reset to layer 0
         if (lEnd > numLayers || rStart < 0) {
            rStart += (lStart + 1) * recurBatchSize;

            lStart = 0;
            lEnd = 1;
         }

         // Off the right, step up
         while (rStart >= seqLength && lEnd <= numLayers) {
            lStart++;
            lEnd++;

            rStart -= recurBatchSize;
         }


         // Over the top or off the left, done!
         if (lEnd > numLayers || rStart < 0) {
            break;
         }
      }

      rEnd = rStart + recurBatchSize;
      if (rEnd > seqLength) rEnd = seqLength;

      for (int layer = lStart; layer < lEnd; layer++) {
         cublasErrCheck(cublasSetStream(handle, stream_i[layer]));

         for (int i = rStart; i < rEnd; i++) {
            if (layer > 0) {
               cudaErrCheck(cudaStreamWaitEvent(stream_i[layer], events_h[layer - 1][i], 0));
               cudaErrCheck(cudaEventDestroy(events_h[layer - 1][i]));
            }
         }

         // Optimization 1
         if (group_gemm) {
            cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        4 * hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                        &alpha,
                        &T_f[layer * 8 * hiddenSize * hiddenSize],
                        transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                        i_data + rStart * numElements + layer * seqLength * numElements,
                        hiddenSize,
                        &beta,
                        tmp_i + 4 * rStart * numElements,
                        4 * hiddenSize));
         }
         else {
            for (int igemm =0; igemm < 4; igemm++) {
               cublasErrCheck(cublasSgemm(handle,
                        transa, transb,
                        hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                        &alpha,
                        &T_f[layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize],
                        transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                        i_data + rStart * numElements + layer * seqLength * numElements,
                        hiddenSize,
                        &beta,
                        tmp_i + 4 * rStart * numElements + igemm * hiddenSize,
                        4 * hiddenSize));
            }
         }

         for (int i = rStart; i < rEnd; i++) {
            cudaErrCheck(cudaEventCreate(&events_i[layer][i], cudaEventDisableTiming));
            cudaErrCheck(cudaEventRecord(events_i[layer][i], stream_i[layer]));
         }

         for (int i = rStart; i < rEnd; i++) {
            cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
            // Optimization 1
            if (group_gemm) {
               cublasErrCheck(cublasSgemm(handle,
                           transa, transb,
                           4 * hiddenSize, miniBatch, hiddenSize,
                           &alpha,
                           &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize],
                           transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                           h_data + i * numElements + layer * (seqLength + 1) * numElements,
                           hiddenSize,
                           &beta,
                           tmp_h + 4 * layer * numElements,
                           4 * hiddenSize));
            }
            else {
               for (int igemm =0; igemm < 4; igemm++) {
                  cublasErrCheck(cublasSgemm(handle,
                              transa, transb,
                              hiddenSize, miniBatch, hiddenSize,
                              &alpha,
                              &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize],
                              transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                              h_data + i * numElements + layer * (seqLength + 1) * numElements,
                              hiddenSize,
                              &beta,
                              tmp_h + 4 * layer * numElements + igemm * hiddenSize,
                              4 * hiddenSize));
               }
            }

            cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_i[layer][i], 0));
            cudaErrCheck(cudaEventDestroy(events_i[layer][i]));

            // Optimization 3
            if (fuse_pw) {
               dim3 blockDim;
               dim3 gridDim;

               blockDim.x = 256;
               gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

               mi_elementWise_fp <<< gridDim, blockDim , 0, stream_h[layer] >>>
                      (hiddenSize, miniBatch,
                       tmp_h + 4 * layer * numElements,
                       tmp_i + 4 * i * numElements,
                       bias + 8 * layer * hiddenSize,
                       alpha_data + 4 * layer * hiddenSize,
                       beta1_data + 4 * layer * hiddenSize,
                       beta2_data + 4 * layer * hiddenSize,
                       TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                       h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       i_data + i * numElements + (layer + 1) * seqLength * numElements,
                       c_data + i * numElements + layer * (seqLength + 1) * numElements,
                       c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       TRAINING);
               cudaErrCheck(cudaGetLastError());
            }
            else {
               LSTM_elementwise_unfused(hiddenSize, miniBatch,
                       tmp_h + 4 * layer * numElements,
                       tmp_i + 4 * i * numElements,
                       bias + 8 * layer * hiddenSize,
                       TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                       h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       i_data + i * numElements + (layer + 1) * seqLength * numElements,
                       c_data + i * numElements + layer * (seqLength + 1) * numElements,
                       c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       TRAINING,
                       stream_h[layer]);
            }
            if (layer != numLayers - 1) {
               cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
               cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer]));
            }
         }
      }
   }
   cudaErrCheck(cudaEventRecord(stop));
   cudaErrCheck(cudaEventSynchronize(stop));
   cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));

   cudaErrCheck(cudaDeviceSynchronize());
   cudaErrCheck(cudaDeviceSynchronize());

   if (TRAINING) cudaErrCheck(cudaFree(linearGates));

   for (int i = 0; i < numLayers; i++) {
      if (stream_i[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_i[i]));
      if (stream_h[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_h[i]));
   }

   free(stream_i);
   free(stream_h);

   for (int i = 0; i < numLayers; i++) {
      free(events_i[i]);
      free(events_h[i]);
   }
   free(events_i);
   free(events_h);

   return { i, h, c, T_, bias_ };
}

