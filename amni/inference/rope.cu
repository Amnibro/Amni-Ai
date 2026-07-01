#include <hip/hip_runtime.h>
#include <cstdint>
__device__ __forceinline__ float bf2f(uint16_t b){uint32_t u=((uint32_t)b)<<16;float f;__builtin_memcpy(&f,&u,4);return f;}
__device__ __forceinline__ uint16_t f2bf(float v){uint32_t u;__builtin_memcpy(&u,&v,4);uint32_t lsb=(u>>16)&1u;u+=0x7fffu+lsb;return (uint16_t)(u>>16);}
__global__ void rope_k(const uint16_t* __restrict__ q,const float* __restrict__ cs,const float* __restrict__ sn,uint16_t* __restrict__ out,int D,int R){
  int h=blockIdx.x;int i=threadIdx.x;int half=R>>1;
  const uint16_t* qh=q+(size_t)h*D;uint16_t* oh=out+(size_t)h*D;
  if(i<half){
    float a=bf2f(qh[i]),b=bf2f(qh[i+half]),c=cs[i],s=sn[i];
    oh[i]=f2bf(a*c-b*s);oh[i+half]=f2bf(b*c+a*s);
  } else if(i>=R && i<D){ oh[i]=qh[i]; }
}
extern "C" __declspec(dllexport) void launch_rope(const void* q,const void* cs,const void* sn,void* out,int NH,int D,int R,void* st){
  rope_k<<<dim3(NH),dim3(D),0,(hipStream_t)st>>>((const uint16_t*)q,(const float*)cs,(const float*)sn,(uint16_t*)out,D,R);
}
