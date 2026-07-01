#include <hip/hip_runtime.h>
#include <cstdint>
#define TB 256
extern "C" __device__ float __ocml_rsqrt_f32(float);
__device__ __forceinline__ float bf2f(uint16_t b){uint32_t u=((uint32_t)b)<<16;float f;__builtin_memcpy(&f,&u,4);return f;}
__device__ __forceinline__ uint16_t f2bf(float v){uint32_t u;__builtin_memcpy(&u,&v,4);uint32_t lsb=(u>>16)&1u;u+=0x7fffu+lsb;return (uint16_t)(u>>16);}
__global__ void rmsnorm_k(const uint16_t* __restrict__ x,const uint16_t* __restrict__ w,uint16_t* __restrict__ y,int H,float eps,float g1){
  int tid=threadIdx.x;float acc=0.f;
  for(int i=tid;i<H;i+=TB){float v=bf2f(x[i]);acc+=v*v;}
  __shared__ float sd[TB];sd[tid]=acc;__syncthreads();
  for(int s=TB/2;s>0;s>>=1){if(tid<s)sd[tid]+=sd[tid+s];__syncthreads();}
  __shared__ float rs;
  if(tid==0)rs=__ocml_rsqrt_f32(sd[0]/(float)H+eps);
  __syncthreads();
  float r=rs;
  for(int i=tid;i<H;i+=TB){float v=bf2f(x[i])*r;float g=g1+bf2f(w[i]);y[i]=f2bf(v*g);}
}
extern "C" __declspec(dllexport) void launch_rmsnorm(const void* x,const void* w,void* y,int H,float eps,float g1,void* st){
  rmsnorm_k<<<dim3(1),dim3(TB),0,(hipStream_t)st>>>((const uint16_t*)x,(const uint16_t*)w,(uint16_t*)y,H,eps,g1);
}
