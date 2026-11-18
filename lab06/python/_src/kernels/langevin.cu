/* Langevin Euler-Maruyama kernels — compiled by CuPy RawModule and PyCUDA SourceModule.
 * extern "C" linkage for name lookup from Python host code.
 */
#include <curand_kernel.h>

extern "C" {

__global__ void density_kernel(
    float* __restrict__ x_out,
    int n,
    int n_steps,
    float sqrt_dt,
    unsigned long long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandStatePhilox4_32_10_t state;
    for (int traj = id; traj < n; traj += stride)
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
    extern __shared__ unsigned int shist[];
    int tid = threadIdx.x;

    if (tid < (unsigned)bins)
    {
        shist[tid] = 0u;
    }
    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float inv_w = (float)bins / (xmax - xmin);

    for (int i = id; i < n; i += stride)
    {
        float v = x[i];
        if (v < xmin || v >= xmax)
        {
            continue;
        }
        int bin = (int)((v - xmin) * inv_w);
        if (bin < 0)
        {
            bin = 0;
        }
        if (bin >= bins)
        {
            bin = bins - 1;
        }
        atomicAdd(&shist[bin], 1u);
    }
    __syncthreads();

    if (tid < bins)
    {
        unsigned int val = shist[tid];
        if (val > 0u)
        {
            atomicAdd(&hist[tid], val);
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
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

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
    extern __shared__ double shmem[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    double local_count = 0.0;
    for (int i = id; i < n; i += stride)
    {
        float v = in[i];
        if (v > 0.0f)
        {
            local_sum += (double)v;
            local_count += 1.0;
        }
    }
    shmem[tid] = local_sum;
    shmem[tid + nthreads] = local_count;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shmem[tid] += shmem[tid + s];
            shmem[tid + nthreads] += shmem[tid + nthreads + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(sum_out, shmem[0]);
        atomicAdd(count_out, (unsigned int)shmem[nthreads]);
    }
}

} /* extern "C" */