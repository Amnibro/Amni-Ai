#include <hip/hip_runtime.h>
#include <cstdint>
extern "C" __device__ float __ocml_exp_f32(float);
__device__ __forceinline__ float bf2f(uint16_t b){uint32_t u=((uint32_t)b)<<16;float f;__builtin_memcpy(&f,&u,4);return f;}
__device__ __forceinline__ uint16_t f2bf(float v){uint32_t u;__builtin_memcpy(&u,&v,4);uint32_t lsb=(u>>16)&1u;u+=0x7fffu+lsb;return (uint16_t)(u>>16);}
__global__ void attn_mq_k(const uint16_t* __restrict__ Q,const uint16_t* __restrict__ Kc,const uint16_t* __restrict__ Vc,uint16_t* __restrict__ out,int NQ,int NH,int NKV,int D,int Lc,int Ltot,float scale,int win){
  int h=blockIdx.x;int i=blockIdx.y;int d=threadIdx.x;int g=NH/NKV;int kv=h/g;
  int qpos=Lc+i;int jstart=(win>0&&win<=qpos)?(qpos-win+1):0;
  const uint16_t* qh=Q+((size_t)i*NH+h)*D;
  const uint16_t* Kk=Kc+(size_t)kv*Ltot*D;const uint16_t* Vk=Vc+(size_t)kv*Ltot*D;
  float qd=bf2f(qh[d]);
  __shared__ float red[256];__shared__ float mS,lS,corrS,pS;
  if(d==0){mS=-1e30f;lS=0.f;}
  float acc=0.f;__syncthreads();
  for(int j=jstart;j<=qpos;j++){
    red[d]=qd*bf2f(Kk[(size_t)j*D+d]);__syncthreads();
    for(int s=D>>1;s>0;s>>=1){if(d<s)red[d]+=red[d+s];__syncthreads();}
    if(d==0){float sc=red[0]*scale;float mn=(mS>sc)?mS:sc;corrS=__ocml_exp_f32(mS-mn);pS=__ocml_exp_f32(sc-mn);lS=lS*corrS+pS;mS=mn;}
    __syncthreads();
    acc=acc*corrS+pS*bf2f(Vk[(size_t)j*D+d]);__syncthreads();
  }
  out[((size_t)i*NH+h)*D+d]=f2bf(acc/lS);
}
extern "C" __declspec(dllexport) void launch_attn_mq(const void* Q,const void* Kc,const void* Vc,void* out,int NQ,int NH,int NKV,int D,int Lc,float scale,int win,void* st){
  attn_mq_k<<<dim3(NH,NQ),dim3(D),0,(hipStream_t)st>>>((const uint16_t*)Q,(const uint16_t*)Kc,(const uint16_t*)Vc,(uint16_t*)out,NQ,NH,NKV,D,Lc,Lc+NQ,scale,win);
}
