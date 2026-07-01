#include <hip/hip_runtime.h>
#include <cstdint>
extern "C" __device__ float __ocml_exp_f32(float);
__device__ __forceinline__ float bf2f(uint16_t b){uint32_t u=((uint32_t)b)<<16;float f;__builtin_memcpy(&f,&u,4);return f;}
__device__ __forceinline__ uint16_t f2bf(float v){uint32_t u;__builtin_memcpy(&u,&v,4);uint32_t lsb=(u>>16)&1u;u+=0x7fffu+lsb;return (uint16_t)(u>>16);}
__global__ void attn_k(const uint16_t* __restrict__ q,const uint16_t* __restrict__ K,const uint16_t* __restrict__ V,uint16_t* __restrict__ out,int NH,int NKV,int D,int L,float scale,int win){
  int h=blockIdx.x;int d=threadIdx.x;int g=NH/NKV;int kv=h/g;
  const uint16_t* qh=q+(size_t)h*D;const uint16_t* Kk=K+(size_t)kv*L*D;const uint16_t* Vk=V+(size_t)kv*L*D;
  float qd=bf2f(qh[d]);
  int jstart=(win>0&&win<L)?(L-win):0;
  __shared__ float red[256];__shared__ float mS,lS,corrS,pS;
  if(d==0){mS=-1e30f;lS=0.f;}
  float acc=0.f;__syncthreads();
  for(int j=jstart;j<L;j++){
    red[d]=qd*bf2f(Kk[(size_t)j*D+d]);__syncthreads();
    for(int s=D>>1;s>0;s>>=1){if(d<s)red[d]+=red[d+s];__syncthreads();}
    if(d==0){float sc=red[0]*scale;float mn=(mS>sc)?mS:sc;corrS=__ocml_exp_f32(mS-mn);pS=__ocml_exp_f32(sc-mn);lS=lS*corrS+pS;mS=mn;}
    __syncthreads();
    acc=acc*corrS+pS*bf2f(Vk[(size_t)j*D+d]);
    __syncthreads();
  }
  out[(size_t)h*D+d]=f2bf(acc/lS);
}
extern "C" __declspec(dllexport) void launch_attn(const void* q,const void* K,const void* V,void* out,int NH,int NKV,int D,int L,float scale,int win,void* st){
  attn_k<<<dim3(NH),dim3(D),0,(hipStream_t)st>>>((const uint16_t*)q,(const uint16_t*)K,(const uint16_t*)V,(uint16_t*)out,NH,NKV,D,L,scale,win);
}
