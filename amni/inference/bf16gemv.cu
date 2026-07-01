#include <hip/hip_runtime.h>
#include <cstdint>
#define TB 128
__device__ __forceinline__ float bf2f(uint16_t b){uint32_t u=((uint32_t)b)<<16;float f;__builtin_memcpy(&f,&u,4);return f;}
__device__ __forceinline__ uint16_t f2bf(float v){uint32_t u;__builtin_memcpy(&u,&v,4);uint32_t lsb=(u>>16)&1u;u+=0x7fffu+lsb;return (uint16_t)(u>>16);}
__global__ void bf16_gemv_k(const uint16_t* __restrict__ W,const uint16_t* __restrict__ x,uint16_t* __restrict__ y,int OUT,int IN){
  int o=blockIdx.x;int tid=threadIdx.x;
  const uint4* wr=(const uint4*)(W+(size_t)o*IN);
  const uint4* xr=(const uint4*)x;
  int n4=IN>>3;float acc=0.f;
  for(int i=tid;i<n4;i+=TB){
    uint4 wv=wr[i];uint4 xv=xr[i];
    uint32_t wq[4]={wv.x,wv.y,wv.z,wv.w};uint32_t xq[4]={xv.x,xv.y,xv.z,xv.w};
    #pragma unroll
    for(int q=0;q<4;q++){uint32_t wu=wq[q],xu=xq[q];acc+=bf2f(wu&0xFFFF)*bf2f(xu&0xFFFF)+bf2f(wu>>16)*bf2f(xu>>16);}
  }
  __shared__ float sd[TB];sd[tid]=acc;__syncthreads();
  for(int s=TB/2;s>0;s>>=1){if(tid<s)sd[tid]+=sd[tid+s];__syncthreads();}
  if(tid==0)y[o]=f2bf(sd[0]);
}
extern "C" __declspec(dllexport) void launch_bf16_gemv(const void* W,const void* x,void* y,int OUT,int IN,void* st){
  bf16_gemv_k<<<dim3(OUT),dim3(TB),0,(hipStream_t)st>>>((const uint16_t*)W,(const uint16_t*)x,(uint16_t*)y,OUT,IN);
}
