#include <hip/hip_runtime.h>
#include <cstdint>
extern "C" __device__ float __ocml_exp_f32(float);
extern "C" __device__ float __ocml_tanh_f32(float);
__device__ __forceinline__ float bf2f(uint16_t b){uint32_t u=((uint32_t)b)<<16;float f;__builtin_memcpy(&f,&u,4);return f;}
__device__ __forceinline__ uint16_t f2bf(float v){uint32_t u;__builtin_memcpy(&u,&v,4);uint32_t lsb=(u>>16)&1u;u+=0x7fffu+lsb;return (uint16_t)(u>>16);}
__global__ void swiglu_k(const uint16_t* __restrict__ g,const uint16_t* __restrict__ u,uint16_t* __restrict__ y,int I,int mode){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<I){
    float gv=bf2f(g[idx]),uv=bf2f(u[idx]),a;
    if(mode==0) a=gv/(1.0f+__ocml_exp_f32(-gv));
    else{float x=gv;float inner=0.7978845608028654f*(x+0.044715f*x*x*x);a=0.5f*x*(1.0f+__ocml_tanh_f32(inner));}
    y[idx]=f2bf(a*uv);
  }
}
extern "C" __declspec(dllexport) void launch_swiglu(const void* g,const void* u,void* y,int I,int mode,void* st){
  int TB=256;swiglu_k<<<dim3((I+TB-1)/TB),dim3(TB),0,(hipStream_t)st>>>((const uint16_t*)g,(const uint16_t*)u,(uint16_t*)y,I,mode);
}
