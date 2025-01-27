#include <cuda.h>
#include <cub/cub.cuh>

//针对小规模的二维张量
template <typename T, typename Tind>
__global__ void gather1(T *data, Tind *indexx, T *out_put, int M,int N, int m,int n, int axis)
{
  int index=threadIdx.x+blockDim.x*blockIdx.x;
  if(axis==0)
  {
    if(index >= m * n*N)
      return;
    int row=(index%(n*N))/N;
    int col=(index%(n*N))%N;
    int blo=index/(n*N);
    out_put[index]=data[indexx[blo*n+row]*N+col];
    //out_put[index]=data[indexx[index/(n*N)*n+(index%(n*N))/N]*N+(index%(n*N))%N];
  }
  if(axis==1)
  {    
  if(index >= M * m* n)
    return;
  int blo =index/(m*n);
  int row =(index%(m*n))/n;
  int col =(index%(m*n))%n;
  out_put[index]=data[blo*N+indexx[row*n+col]];
  }
}
//针对较大规模的二维张量
template <typename T, typename Tind>
__global__ void gather_big(T *data, Tind *indexx, T *out_put, int M,int N, int m,int n, int axis)
{
  int index=threadIdx.x+blockDim.x*blockIdx.x;
  int tmp=2*index;
  if(axis==0)
  {
    if(index >= m * n*N)
      return;
    for(int i=0;i<2;i++)
    {
      int index_x=tmp+i;
      int row=(index_x%(n*N))/N;
      int col=(index_x%(n*N))%N;
      int blo=index_x/(n*N);
      out_put[index_x]=data[indexx[blo*n+row]*N+col];
    }

  }
  if(axis==1)
  {    
  if(index >= M * m* n)
    return;
  int blo =index/(m*n);
  int row =(index%(m*n))/n;
  int col =(index%(m*n))%n;
  out_put[index]=data[blo*N+indexx[row*n+col]];
  }
}

//接下来这个尝试处理任意维度的gather,但是性能还未优化。
//M是输入的张量维数，N是索引张量的维数。
//total_out是输出的个数，total_data是输入的个数，total_ind是索引的个数。
template <typename T, typename Tind>
__global__ void high(T *data, Tind *indexx, T *out_put,Tind *array, int N,int M,int total_out,int total_ind,int total_data, int j,int axis)
{
  int index=threadIdx.x+blockDim.x*blockIdx.x;
  if(index >=total_out)
    return;
//   printf("total_out =%d\n ,total_ind=%d\n,total_data=%d\n",total_out,total_ind,total_data);
    int index_tmp = index;
    int k = 0, a = 0, q = 0;

    for (int i = 0; i < N + M - 1; i++) {
        total_out /= array[i];
        a = index_tmp / total_out;
        index_tmp = index_tmp % (total_out);

        if (i >= axis && i < axis + N) {
            if (i == axis) {
                total_data /= j;
            }
            total_ind /= array[i];
            q = q + a * total_ind;
            if(i==axis+N-1){
              k=k+indexx[q]*total_data;
           }
        } else {
            total_data /= array[i];
            k = k + a * total_data;
        }
    }
    out_put[index] = data[k];
    }

extern "C" void gather_my(void const *data, void const *indexx, void *out_put, int M,int N, int m,int n, int axis)
{

    dim3 block_dim;
    dim3 grid_dim;
    if(M*m*n<1024&&m*n*N<1024)
    {
      block_dim = dim3(16, 1, 1); // 赋值而不是重新声明
      grid_dim = dim3(64, 1, 1);
    }
    else
    {
      block_dim=dim3(256, 1, 1);
      grid_dim=dim3((m*n*N/512)+1, 1, 1);
    }
    gather_big<float, uint64_t><<<grid_dim, block_dim>>>(
        reinterpret_cast<float*>(const_cast<void*>(data)),
        reinterpret_cast<uint64_t*>(const_cast<void*>(indexx)),
        reinterpret_cast<float*>(out_put),
        M, N, m, n, axis
    );
}

extern "C" void gather_my_16(void const *data, void const *indexx, void *out_put, int M,int N, int m,int n, int axis)
{

    dim3 block_dim;
    dim3 grid_dim;
    if(M*m*n<1024&&m*n*N<1024)
    {
      block_dim = dim3(64, 1, 1); // 赋值而不是重新声明
      grid_dim = dim3(128, 1, 1);
    }
    else
    {
      block_dim=dim3(128, 1, 1);
      grid_dim=dim3((m*n*N/128)+1, 1, 1);
    }
    gather1<half, uint64_t><<<grid_dim, block_dim>>>(
        reinterpret_cast<half*>(const_cast<void*>(data)),
        reinterpret_cast<uint64_t*>(const_cast<void*>(indexx)),
        reinterpret_cast<half*>(out_put),
        M, N, m, n, axis
    );
}

extern "C" void gather_dim_h(void const *data, void const *indexx,
             void *out_put, void *array,int N,int M, int total_out,int total_ind,int total_data, int j,int axis)
{

    dim3 block_dim;
    dim3 grid_dim;
    block_dim=dim3(64, 1, 1);
    grid_dim=dim3(64, 1, 1);
    high<float, uint64_t><<<grid_dim, block_dim>>>(
        reinterpret_cast<float*>(const_cast<void*>(data)),
        reinterpret_cast<uint64_t*>(const_cast<void*>(indexx)),
        reinterpret_cast<float*>(out_put),
        reinterpret_cast<uint64_t*>(const_cast<void*>(array)),
        N,M,total_out,total_ind,total_data,j,axis
    );
}

extern "C" void gather_dim_h_16(void const *data, void const *indexx,
             void *out_put, void *array,int N,int M, int total_out,int total_ind,int total_data, int j,int axis)
{

    dim3 block_dim;
    dim3 grid_dim;
    block_dim=dim3(64, 1, 1);
    grid_dim=dim3(64, 1, 1);
    high<half, uint64_t><<<grid_dim, block_dim>>>(
        reinterpret_cast<half*>(const_cast<void*>(data)),
        reinterpret_cast<uint64_t*>(const_cast<void*>(indexx)),
        reinterpret_cast<half*>(out_put),
        reinterpret_cast<uint64_t*>(const_cast<void*>(array)),
        N,M,total_out,total_ind,total_data,j,axis
    );
}
