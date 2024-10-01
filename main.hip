// c++20

#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
// #include <cmath>
#include <vector>
#include <chrono>
#include <cuda_profiler_api.h>
#include <hipblas/hipblas.h>

using namespace std::chrono;

// Helpers for error checking:
#define CHECK_RET_CODE(call, ret_code)                       \
  {                                                          \
    if ((call) != ret_code)                                  \
    {                                                        \
      std::cout << "Failed in call: " << #call << std::endl; \
      std::abort();                                          \
    }                                                        \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess);


// Reference CPU c++ implementation from
// https://github.com/ROCm/rocm-examples/blob/0fdfd7f6e7e76199cd795c02ed592569ac96d344/Common/example_utils.hpp#L216C1-L245C2
//
/// \brief Multiply an $A$ matrix ($m \times k$) with a $B$ matrix ($k \times n$) as:
/// $C := \alpha \cdot A \cdot B + \beta \cdot C$
template<typename T>
void multiply_matrices(T        alpha,
                       T        beta,
                       int      m,
                       int      n,
                       int      k,
                       const T* A,
                       int      stride1_a,
                       int      stride2_a,
                       const T* B,
                       int      stride1_b,
                       int      stride2_b,
                       T*       C,
                       int      stride_c,
                       T*       D)
{
    for(int i1 = 0; i1 < m; ++i1)
    {
        for(int i2 = 0; i2 < n; ++i2)
        {
            T t = T(0.0);
            for(int i3 = 0; i3 < k; ++i3)
            {
                t += A[i1 * stride1_a + i3 * stride2_a] * B[i3 * stride1_b + i2 * stride2_b];
            }
            C[i1 + i2 * stride_c] = beta * C[i1 + i2 * stride_c] + alpha * t;
          C[i1 + i2 * stride_c] += D[i1 + i2 * stride_c];
        }      
    }
}




// This defines the kernel that can be launched on GPU:
void __global__ UnoptimisedAXBAC(const std::size_t dimA1, const std::size_t dimA2, const float *A,
                                 const std::size_t dimB1, const std::size_t dimB2, const float *B,
                                 const std::size_t dimC1, const std::size_t dimC2, const float *C,
                                 float *D)
{
  for (std::size_t iter = 0; iter < 10; iter++)
  {

    // printf("(int)(threadIdx.x + blockIdx.x * blockDim.x) %d\n", (int)(threadIdx.x + blockIdx.x * blockDim.x));

    const std::size_t idx = (size_t)(threadIdx.x + blockIdx.x * blockDim.x);

    // printf("local threadidx %d blockidx %d blockdim %d  ==> idx %d \n",threadIdx.x, blockIdx.x, blockDim.x, (int)idx);

    // multiply i'th row from A with j'th column from B and add element i,j from C
    int row = (int)(idx % dimC1);
    int col = (int)(idx / dimC2);

    // printf("idx %d row %d col %d\n", idx, row, col);

    float cellsum = 0.0;
    // Size of A is dimA1 rows and dimA2 columns.
    // Size of B is dimB1 rows and dimB2 columns.
    // With row first in
    for (int i = 0; i < dimA2; i++)
    {
      // Iterate through the row in A and column in B:
      int cellA = row * dimA1 + i;
      int cellB = col + i * dimB2;

      cellsum += (A[cellA] * B[cellB]);

    }
    D[idx] = cellsum + (float)(C[int(idx)]);
  }
}



void __global__ moreBlocksAXBACwithSharedMemory(const std::size_t dimA1, const std::size_t dimA2, const float *A,
                                                const std::size_t dimB1, const std::size_t dimB2, const float *B,
                                                const std::size_t dimC1, const std::size_t dimC2, const float *C,
                                                float *D, int gemmBlockSize)
{

  extern __shared__ float cache[];
  float *gemmCacheA = cache;                                               
  float *gemmCacheB = (float *)&gemmCacheA[gemmBlockSize * gemmBlockSize]; 
  float *cellSums = (float *)&gemmCacheB[gemmBlockSize * gemmBlockSize];   

  for (std::size_t iter = 0; iter < 1; iter++)
  {
    const std::size_t idx0 = (size_t)(threadIdx.x + blockIdx.x * blockDim.x);

    for (int idx = (int)idx0; idx < dimC1*dimC2 ; idx += blockDim.x * gridDim.x ) {

      // multiply i'th row from A with j'th column from B and add element i,j from C
      // int row = (int)(idx % dimC1);
      // int col = (int)(idx / dimC2);

      for (int Sidx = threadIdx.x; Sidx < gemmBlockSize * gemmBlockSize; Sidx += blockDim.x)
      {
        cellSums[Sidx] = 0;
      }
      __syncthreads();

      // Size of A is dimA1 rows and dimA2 columns.
      // Size of B is dimB1 rows and dimB2 columns.

      // Let's do the blocks:
      //
      //   - -  - -          - - - - -       - - B -
      //   - - AB -          A A A A A       - - B -
      //   - -  - -    =     - - - - -   *   - - B -
      //   - -  - -          - - - - -       - - B -
      //

      int A_block_row = (idx / dimA1) / gemmBlockSize; // floor
      int B_block_col = (idx % dimB2) / gemmBlockSize; // floor

      for (int blockA = 0; blockA < dimA2; blockA += gemmBlockSize)
      {
        // A row of blocks:
        // - - - - -
        // A A A A A
        // - - - - -
        // - - - - -

        for (int Aidx = threadIdx.x; Aidx < gemmBlockSize * gemmBlockSize; Aidx += blockDim.x)
        {
          gemmCacheA[Aidx] = A[A_block_row * gemmBlockSize + blockA * dimA1 + Aidx];
        }
        __syncthreads();

        for (int blockB = 0; blockB < dimB2; blockB += gemmBlockSize)
        {
          //  B column of blocks:
          //   - - B -
          //   - - B -
          //   - - B -
          //   - - B -

          for (int Bidx = threadIdx.x; Bidx < gemmBlockSize * gemmBlockSize; Bidx += blockDim.x)
          {
            // This thread is operating on B_block_col
            // So index is blockB * gemmBlocksize
            gemmCacheB[Bidx] = B[blockB * dimB1 + B_block_col * gemmBlockSize + Bidx];
          }
          __syncthreads();

          for (int Sidx = threadIdx.x; Sidx < gemmBlockSize * gemmBlockSize; Sidx += blockDim.x)
          {
            int sum_col = Sidx % gemmBlockSize;
            int sum_row = Sidx / gemmBlockSize; // floor

            for (int rowcol = 0; rowcol < gemmBlockSize; rowcol++)
            {
              // multiply i'th row from blockA with j'th column from blockB and add element i,j from C
              cellSums[sum_row * gemmBlockSize + sum_col] += gemmCacheA[sum_row * gemmBlockSize + rowcol] * gemmCacheB[rowcol * gemmBlockSize + sum_row];
            }
          }
        }
      }
      __syncthreads();

      for (int Sidx = threadIdx.x; Sidx < gemmBlockSize * gemmBlockSize; Sidx += blockDim.x)
      {
        D[Sidx] = cellSums[Sidx] + (float)(C[Sidx + blockIdx.x * blockDim.x]);
      }
    }

    // if (threadIdx.x < 10) {
    //   if (blockIdx.x < 10) {
    //     printf("thread %d from block %d exiting", threadIdx.x, blockIdx.x);
    //   }
    // }
    __syncthreads();
  }
}

int main(int argc, char *argv[])
{

  // Check our hardware:

  int deviceCount;
  hipGetDeviceCount(&deviceCount);

  hipDeviceProp_t deviceProp;

  for (int i = 0; i < deviceCount; ++i)
  {

    hipGetDeviceProperties(&deviceProp, i);

    std::cout << "Device " << i << " properties:" << std::endl;
    std::cout << "  Device name: " << deviceProp.name << std::endl;
    std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads dimensions: ("
              << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max grid dimensions: ("
              << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "  L2 size: " << deviceProp.l2CacheSize << std::endl;
    std::cout << std::endl;
  }

  int max_threads = (int)deviceProp.maxThreadsDim[0];
  int max_blocks = (int)deviceProp.maxGridSize[0];
  int max_threads_per_block = (int)deviceProp.maxThreadsPerBlock;

  // Create a pinned memory for a float array, both host and device can use it

  // Size of A is dimA1 rows and dimA2 columns.
  // Size of B is dimB1 rows and dimB2 columns.
  // dimA2 and dimB1 must be the same!
  // output size is dimA1 x dimB2

  constexpr std::size_t dimA1{640};
  constexpr std::size_t dimA2{896};
  constexpr std::size_t dimB2{704};

  //constexpr std::size_t dimA1{16384};
  //constexpr std::size_t dimA2{28672};

  //constexpr std::size_t dimB2{24576};


  constexpr std::size_t dimB1{dimA2};

  constexpr std::size_t dimC1{dimA1};
  constexpr std::size_t dimC2{dimB2};



  // Allocate memory for fp32 arrays in HOST memory
  // using hipHostMalloc for zero-copy access to values!

  float *A{};
  float *B{};
  float *C{};
  float *D{};
  float *ref{};

  std::cout << "Allocating memory on host" << std::endl;

  HIP_CHECK(hipHostMalloc(&A, dimA1 * dimA2 * sizeof(float)));
  HIP_CHECK(hipHostMalloc(&B, dimB1 * dimB2 * sizeof(float)));
  HIP_CHECK(hipHostMalloc(&C, dimC1 * dimC2 * sizeof(float)));
  HIP_CHECK(hipHostMalloc(&D, dimC1 * dimC2 * sizeof(float)));
  HIP_CHECK(hipHostMalloc(&ref, dimC1 * dimC2 * sizeof(float)));


  // Allocate memory for fp32 arrays in GPU memory
  // using hipMallocManaged (why managed? I already forgot!)

  float *A_{};
  float *B_{};
  float *C_{};
  float *D_{};
  
  std::cout << "Allocating memory on GPU" << std::endl;

  HIP_CHECK(hipMallocManaged(&A_, dimA1 * dimA2 * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&B_, dimB1 * dimB2 * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&C_, dimC1 * dimC2 * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&D_, dimC1 * dimC2 * sizeof(float)));



  // std::cout << "Filling fp32 arrays with random on HOST:" << std::endl;

  // // Create some data
  // constexpr std::float_t min_value{-0.01};
  // constexpr std::float_t max_value{0.05};

  // std::random_device rd;
  // std::mt19937 gen(rd());

  // // Create a distribution for float
  // std::uniform_real_distribution<float> dis(min_value, max_value);

  // for (std::size_t i = 0; i < dimA1 * dimA2; ++i)
  // {
  //   // x[i] = static_cast<float>(dis(gen)); //
  //   A[i] = static_cast<float>(dis(gen));
  // }

  // std::cout << "some values of A:";
  // for (std::size_t i = 0; i < 6; ++i)
  // {
  //   std::cout << A[i] << " ";
  // }
  // std::cout << std::endl;

  // for (std::size_t i = 0; i < dimB1 * dimB2; ++i)
  // {
  //   // x[i] = static_cast<float>(dis(gen)); //
  //   B[i] = static_cast<float>(dis(gen));
  // }

  // for (std::size_t i = 0; i < dimC1 * dimC2; ++i)
  // {
  //   // x[i] = static_cast<float>(dis(gen)); //
  //   C[i] = static_cast<float>(dis(gen));
  //   D[i] = 0.0;
  // }


  // Define things for timing operations

  auto start = high_resolution_clock::now();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);



  // How fast is the memory transfer? Let's use the slower synchronous copy:

  std::cout << "Time 10 x HipMemcpy: "<< std::endl;

  start = high_resolution_clock::now();

  for (int iter = 0; iter < 10; iter++)
  {
    HIP_CHECK(hipMemcpy(A_, A, dimA1 * dimA2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_, B, dimB1 * dimB2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(C_, C, dimC1 * dimC2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(D_, D, dimC1 * dimC2 * sizeof(float), hipMemcpyHostToDevice));
    hipDeviceSynchronize();
  }
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "   One operation took " << (float)duration.count() / 1000000.0 /10.0 << " seconds on average" << std::endl;



  // The following was for testing hipblas operations but I never got that far, so let's skip these lines:

  const int m = dimA1; //"Number of rows of matrices A_i and C_i");
  const int n = dimB2; //"Number of columns of matrices B_i and C_i");
  const int k = dimB2; //"Number of columns of matrix A_i and rows of B_i");

  const float h_alpha = 1.0;
  const float h_beta = 1.0;

  // Set GEMM operation as identity operation: $op(X) = X$
  const hipblasOperation_t trans_a = HIPBLAS_OP_N;
  const hipblasOperation_t trans_b = HIPBLAS_OP_N;

  int           lda, ldb, ldc;
  int           stride1_a, stride2_a, stride1_b, stride2_b;
  hipblasStride stride_a, stride_b, stride_c;

  // Set up matrix dimension variables.
  if(trans_a == HIPBLAS_OP_N)
  {
      lda       = m;
      stride_a  = hipblasStride(k) * lda;
      stride1_a = 1;
      stride2_a = lda;
  }
  else
  {
      lda       = k;
      stride_a  = hipblasStride(m) * lda;
      stride1_a = lda;
      stride2_a = 1;
  }
  if(trans_b == HIPBLAS_OP_N)
  {
      ldb       = k;
      stride_b  = hipblasStride(n) * ldb;
      stride1_b = 1;
      stride2_b = ldb;
  }
  else
  {
      ldb       = n;
      stride_b  = hipblasStride(k) * ldb;
      stride1_b = ldb;
      stride2_b = 1;
  }
  ldc      = m;
  stride_c = hipblasStride(n) * ldc;




  // And here we're back to our experiment.

  // Let's time the naive matrix multiplication, accessing data directly from
  // global memory:


  // How much compute can we make?
  // gtx 1050 apparently has 640 CUDA cores.
  // rtx 4090 has 16384 CUDA cores.

  std::cout << "number of output cells: " << dimC1 * dimC2 << std::endl;

  constexpr int num_output_cells = dimC1 * dimC2;

  constexpr dim3 number_of_blocks(num_output_cells / 16);
  // constexpr dim3 number_of_threads_per_block(A1 / 640 + 1);
  constexpr dim3 number_of_threads_per_block(16);
  std::size_t blockdim = 16;


  if (dimC1 * dimC2 < 4000000) {
    // Time and launch kernel:

    std::cout << "Time 10 x naive multiplication with zero-copy memory access:" << std::endl;

    start = high_resolution_clock::now();

    hipLaunchKernelGGL(UnoptimisedAXBAC, number_of_blocks, number_of_threads_per_block, blockdim, 0, dimA1, dimA2, A, dimB1, dimB2, B, dimC1, dimC2, C, D); // calls HelloWorld(N, x)
    hipDeviceSynchronize();

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "    Time taken by one iteration: " << (float)duration.count() / 1000000.0 / 10.0 << " seconds on average" << std::endl;
  }



  // Let's try the same accessing the data from global memory:

  std::cout << "Time 10 x naive multiplication with global memory: " << std::endl;

  start = high_resolution_clock::now();

  hipLaunchKernelGGL(UnoptimisedAXBAC, number_of_blocks, number_of_threads_per_block, blockdim, 0, dimA1, dimA2, A_, dimB1, dimB2, B_, dimC1, dimC2, C_, D_); // calls HelloWorld(N, x)
  hipDeviceSynchronize();

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "    Time taken by one iteration " << (float)duration.count() / 1000000.0 / 10.0 << " seconds on average" << std::endl;



  // Let's try the block based approach:
  
  // Define our block size:
  constexpr int gemmBlockSize = 16;

  // define the required shared memory size:
  constexpr std::size_t sharedMemSize = 3 * gemmBlockSize * gemmBlockSize * sizeof(float);

  // Find the amount of blocks we can use based on device shared memory size:
  int max_usable_blocks = (int)deviceProp.l2CacheSize/(int)sharedMemSize;
  std::cout << "There is enough L2 for " << max_usable_blocks << " blocks each using " << (int)sharedMemSize << " shared mem" << std::endl;
  
  // Find the amount of blocks we need to calculate:
  int required_blocks = (dimC1/gemmBlockSize) * (dimC2 / gemmBlockSize);
  
  int using_blocks = min(max_usable_blocks, required_blocks);

  // Set the kernel launch dimensions:
  dim3 number_of_blocks_c( using_blocks  );
  constexpr dim3 number_of_threads_per_block_c(gemmBlockSize*gemmBlockSize);
  constexpr int blockdim_c(gemmBlockSize*gemmBlockSize);

  // Do a final check:
  if ((int)sharedMemSize * (int)number_of_blocks_c.x > deviceProp.l2CacheSize) {
    std::cout << "Shared memory " << deviceProp.l2CacheSize << " is not enough for " << number_of_blocks_c.x << " with shared mem each " << (int)sharedMemSize << std::endl;
  }
  else {

    std::cout << "Time 10 x block based multiplication with shared memory: " << std::endl;

    start = high_resolution_clock::now();
    // cudaProfilerStart();

    // I messed up the hipLaunchKernelGGL function definitions somehow, 
    // and reverted to CUDA syntax of using <<< and >>> for launching the kernel:
    moreBlocksAXBACwithSharedMemory<<<number_of_blocks_c, number_of_threads_per_block_c, sharedMemSize, 0>>>(dimA1, dimA2, A_, dimB1, dimB2, B_, dimC1, dimC2, C_, D_, gemmBlockSize);
    hipDeviceSynchronize();
    // cudaProfilerStop();

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "     Time taken by one iteration on average: " << (float)duration.count() / 1000000.0 /10.0<< " seconds on average" << std::endl;
  }


  // constexpr dim3 number_of_threads_per_block_b(dimB2/32);
  // start = high_resolution_clock::now();
  // cudaProfilerStart();
  // hipLaunchKernelGGL(thirtytwoBlockAXBAC, number_of_blocks, number_of_threads_per_block_b, blockdim, 0, dimA1, dimA2, A_, dimB1, dimB2, B_, dimC1, dimC2, C_, D_);  // calls HelloWorld(N, x)
  // hipDeviceSynchronize();
  // cudaProfilerStop();
  // // Ending time for the clock
  // stop = high_resolution_clock::now();
  // duration = duration_cast<microseconds>(stop - start);
  // std::cout << "Time taken by 10 x thirtytwoBlockAXBAC with global memory: " << (float)duration.count()/1000000.0 << " seconds" << std::endl;


  // std::cout << "some values of A_ after 1000 sync transfers:";
  // for(std::size_t i = 0; i < 6; ++i)
  // {
  //     std::cout << A_[i] << " ";
  // }
  // std::cout << std::endl;

  start = high_resolution_clock::now();

  for (int iter = 0; iter < 10; iter++)
  {
    HIP_CHECK(hipMemcpyAsync(A_, A, dimA1 * dimA2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyAsync(B_, B, dimB1 * dimB2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyAsync(C_, C, dimC1 * dimC2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyAsync(D_, D, dimC1 * dimC2 * sizeof(float), hipMemcpyHostToDevice));
    hipDeviceSynchronize();
  }

  // std::cout << "some values of A_ after 1000 async transfers:";
  // for(std::size_t i = 0; i < 6; ++i)
  // {
  //     std::cout << A_[i] << " ";
  // }
  // std::cout << std::endl;

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "Time taken by 10 x HipMemcpyAsync: " << (float)duration.count() / 1000000.0 /10.0 << " seconds on average" << std::endl;

  // start = high_resolution_clock::now();
  // cudaProfilerStart();
  // hipLaunchKernelGGL(thirtytwoBlockAXBAC, number_of_blocks, number_of_threads_per_block_b, blockdim, 0, dimA1, dimA2, A_, dimB1, dimB2, B_, dimC1, dimC2, C_, D_);  // calls HelloWorld(N, x)
  // hipDeviceSynchronize();
  // cudaProfilerStop();
  // // Ending time for the clock
  // stop = high_resolution_clock::now();
  // duration = duration_cast<microseconds>(stop - start);
  // std::cout << "Time taken by 10 x thirtytwoBlockAXBAC with global memory: " << (float)duration.count()/1000000.0 << " seconds" << std::endl;

  // start = high_resolution_clock::now();
  // hipLaunchKernelGGL(fourInARowAXBAC, number_of_blocks, number_of_threads_per_block, blockdim, 0, dimA1, dimA2, A_, dimB1, dimB2, B_, dimC1, dimC2, C_, D_); // calls HelloWorld(N, x)
  // hipDeviceSynchronize();
  // // Ending time for the clock
  // stop = high_resolution_clock::now();
  // duration = duration_cast<microseconds>(stop - start);
  // std::cout << "Time taken by 10 x fourInARowAXBAC with global memory: " << (float)duration.count() / 1000000.0 << " seconds" << std::endl;

  // start = high_resolution_clock::now();
  // hipLaunchKernelGGL(eightInARowAXBAC, number_of_blocks, number_of_threads_per_block, blockdim, 0, dimA1, dimA2, A_, dimB1, dimB2, B_, dimC1, dimC2, C_, D_); // calls HelloWorld(N, x)
  // hipDeviceSynchronize();
  // // Ending time for the clock
  // stop = high_resolution_clock::now();
  // duration = duration_cast<microseconds>(stop - start);
  // std::cout << "Time taken by 10 x eightInARowAXBAC with global memory: " << (float)duration.count() / 1000000.0 << " seconds" << std::endl;




  start = high_resolution_clock::now();
  multiply_matrices<float>(h_alpha,
                                 h_beta,
                                 m,
                                 n,
                                 k,
                                 A,
                                 stride1_a,
                                 stride2_a,
                                 B,
                                 stride1_b,
                                 stride2_b,
                                 ref,
                                 ldc,
                                 C);


  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "Time taken by CPU reference implementation: " << (float)duration.count() / 1000000.0 << " seconds" << std::endl;




  std::cout << "Dims of D: " << dimC1 << " " << dimC2 << std::endl;
  for (std::size_t i = 0; i < min((int)(dimC1 * dimC2), (int)10); ++i)
  {
    printf("%f ", D[int(i)]);
    if ((i + 1) % dimC2 == 0)
    {
      std::cout << std::endl;
    }
    // D[i] = 0.0;
  }
  std::cout << std::endl;

  HIP_CHECK(hipFree(A_));
  HIP_CHECK(hipFree(B_));
  HIP_CHECK(hipFree(C_));
  HIP_CHECK(hipFree(D_));

  HIP_CHECK(hipHostFree(A));
  HIP_CHECK(hipHostFree(B));
  HIP_CHECK(hipHostFree(C));
  HIP_CHECK(hipHostFree(D));
  return 0;
}
