#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#define TB 128
__global__ void nvfp4_gemv_k(const uint8_t* __restrict__ codes,const __half* __restrict__ scale,const __half* __restrict__ x,__half* __restrict__ y,const float* __restrict__ lut,float ws2,int OUT,int HALF,int NG){
  int o=blockIdx.x; int tid=threadIdx.x;
  const uint4* c4=(const uint4*)(codes+(size_t)o*HALF);
  const __half* srow=scale+(size_t)o*NG;
  int n4=HALF>>4;
  float acc=0.f;
  for(int i=tid;i<n4;i+=TB){
    uint4 v=c4[i];
    int base=i<<4;
    float s0=__half2float(srow[base>>3])*ws2;
    float s1=__half2float(srow[(base+8)>>3])*ws2;
    const __half* xp=x+(base<<1);
    uint32_t w[4]={v.x,v.y,v.z,v.w};
    #pragma unroll
    for(int q=0;q<4;q++){
      uint32_t u=w[q];
      #pragma unroll
      for(int bts=0;bts<4;bts++){
        int j=(q<<2)+bts;
        int b=(u>>(bts<<3))&0xFF;
        float sc=(j<8)?s0:s1;
        float xe=__half2float(xp[2*j]);
        float xo=__half2float(xp[2*j+1]);
        acc+=sc*(lut[b&0xF]*xe+lut[(b>>4)&0xF]*xo);
      }
    }
  }
  __shared__ float sdata[TB];
  sdata[tid]=acc; __syncthreads();
  for(int s=TB/2;s>0;s>>=1){ if(tid<s) sdata[tid]+=sdata[tid+s]; __syncthreads(); }
  if(tid==0) y[o]=__float2half(sdata[0]);
}
extern "C" __declspec(dllexport) void launch_nvfp4_gemv(const void* codes,const void* scale,const void* x,void* y,const void* lut,float ws2,int OUT,int HALF,int NG,void* stream){
  nvfp4_gemv_k<<<dim3(OUT),dim3(TB),0,(hipStream_t)stream>>>((const uint8_t*)codes,(const __half*)scale,(const __half*)x,(__half*)y,(const float*)lut,ws2,OUT,HALF,NG);
}
