/* Langevin Euler-Maruyama kernels — compiled by CuPy RawModule and PyCUDA SourceModule.
 * extern "C" linkage for name lookup from Python host code.
 */
#include <curand_kernel.h>

extern "C" {

__device__ __forceinline__ int clip_int(int value, int lo, int hi)
{
    return min(max(value, lo), hi);
}



// every function -> single thread in a specific block      many blocks working in parallel to each other



__global__ void density_kernel(
    float* __restrict__ x_out,
    int n,
    int n_steps,
    float sqrt_dt,
    unsigned long long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;     // position of thread in grid
    int stride = blockDim.x * gridDim.x;                // whole grid jump

    curandStatePhilox4_32_10_t state;
    for (int traj = id; traj < n; traj += stride)       // traj < n  ->  safety guard
    {
        curand_init(seed, (unsigned long long)traj, 0ULL, &state);
        float x = 0.0f;
        for (int s = 0; s < n_steps; ++s)
        {
            float xi = curand_normal(&state);
            x += sqrt_dt * xi;
        }
        x_out[traj] = x;
    }
}

__global__ void histogram_kernel(
    const float* __restrict__ x,
    int n,
    unsigned int* __restrict__ hist,
    int bins,
    float xmin,
    float xmax)
{
    extern __shared__ unsigned int shist[];             // per block   - only one dynamicly shared
    int tid = threadIdx.x;

    if (tid < bins)
    {
        shist[tid] = 0u;
    }
    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;     // position of thread in grid
    int stride = blockDim.x * gridDim.x;                // whole grid jump
    float inv_w = (float)bins / (xmax - xmin);

    for (int i = id; i < n; i += stride)
    {
        float v = x[i];
        if (v < xmin || v >= xmax)
        {
            continue;
        }
        int bin = (int)((v - xmin) * inv_w);
        bin = clip_int(bin, 0, bins - 1);
        atomicAdd(&shist[bin], 1u);        // other threads in this block may hit the same bin — must be atomic
    }
    __syncthreads();

    if (tid < bins)
    {
        unsigned int val = shist[tid];                   // taking value out of shared mem in block
        if (val > 0u)
        {
            atomicAdd(&hist[tid], val);         // atomic needed -> many blocks are running at the same time
        }
    }
}

__global__ void mfpt_kernel(
    float* __restrict__ exit_t,
    int n,
    float dt,
    float sqrt_dt,
    int max_steps,
    unsigned long long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;     // position of thread in grid
    int stride = blockDim.x * gridDim.x;                // whole grid jump

    curandStatePhilox4_32_10_t state;
    for (int traj = id; traj < n; traj += stride)
    {
        curand_init(seed, (unsigned long long)traj, 0ULL, &state);
        float x = 0.0f;
        float t = 0.0f;
        int steps = 0;
        while (steps < max_steps)
        {
            float xi = curand_normal(&state);
            x += sqrt_dt * xi;
            t += dt;
            ++steps;
            if (x > 1.0f || x < -1.0f)
            {
                exit_t[traj] = t;
                break;
            }
        }
        if (steps >= max_steps)
        {
            exit_t[traj] = -t;
        }
    }
}

__global__ void sum_reduce_kernel(
    const float* __restrict__ in,
    int n,
    double* __restrict__ sum_out,
    unsigned int* __restrict__ count_out)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;     // position of thread in grid
    int stride = blockDim.x * gridDim.x;                // whole grid jump

    double local_sum = 0.0;
    double local_count = 0.0;
    for (int i = id; i < n; i += stride)                // summing into thread registers
    {
        float v = in[i];
        if (v > 0.0f)
        {
            local_sum += (double)v;
            local_count += 1.0;
        }
    }

    extern __shared__ char shmem_bytes[];               // per block   - only one dynamicly shared
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    double* shmem_sum = (double*)(shmem_bytes);
    double* shmem_count = (double*)(shmem_bytes + nthreads * sizeof(double));

    shmem_sum[tid] = local_sum;
    shmem_count[tid] = local_count;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1)          // summing registers between threads in one block
    {
        if (tid < s)                                    // cuts number of active threads by half
        {
            shmem_sum[tid] += shmem_sum[tid + s];
            shmem_count[tid] += shmem_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)                                                               // only single thread adds to global return value
    {
        atomicAdd(sum_out, shmem_sum[0]);                          // other blocks running
        atomicAdd(count_out, (unsigned int)shmem_count[0]);
    }
}

} /* extern "C" */