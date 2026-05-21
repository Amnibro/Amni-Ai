#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#define TILE_H 8
#define TILE_W 8
#define BLOCK_F 16
__global__ void k_conv3x3_s1p1_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ b,
    __half* __restrict__ y,
    int N, int Cin, int Cout, int H, int W){
    int n=blockIdx.z;
    int f_blk=blockIdx.y*BLOCK_F+threadIdx.y;
    int ow=blockIdx.x*(TILE_H*TILE_W)+threadIdx.x;
    if(f_blk>=Cout||ow>=H*W)return;
    int oh=ow/W,owc=ow%W;
    float acc=(b!=nullptr)?(float)b[f_blk]:0.0f;
    for(int c=0;c<Cin;c++){
        const __half* wp=w+((f_blk*Cin+c)*9);
        const __half* xp=x+((n*Cin+c)*H*W);
        #pragma unroll
        for(int kh=0;kh<3;kh++){
            #pragma unroll
            for(int kw=0;kw<3;kw++){
                int hh=oh+kh-1,ww=owc+kw-1;
                if(hh<0||hh>=H||ww<0||ww>=W)continue;
                float xv=(float)xp[hh*W+ww];
                float wv=(float)wp[kh*3+kw];
                acc+=xv*wv;
            }
        }
    }
    y[((n*Cout+f_blk)*H+oh)*W+owc]=(__half)acc;
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int conv3x3_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT void* conv3x3_alloc(size_t n){void* p=nullptr; hipMalloc(&p,n); return p;}
EXPORT void conv3x3_free(void* p){if(p)hipFree(p);}
EXPORT int conv3x3_h2d(void* d, const void* s, size_t n){return hipMemcpy(d,s,n,hipMemcpyHostToDevice)==hipSuccess?0:-1;}
EXPORT int conv3x3_d2h(void* d, const void* s, size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToHost)==hipSuccess?0:-1;}
EXPORT int conv3x3_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int conv3x3_run(
    const void* x, const void* w, const void* b, void* y,
    int N, int Cin, int Cout, int H, int W){
    int pix=H*W;
    int tile=TILE_H*TILE_W;
    dim3 block(tile, BLOCK_F);
    dim3 grid((pix+tile-1)/tile, (Cout+BLOCK_F-1)/BLOCK_F, N);
    hipLaunchKernelGGL(k_conv3x3_s1p1_fp16, grid, block, 0, 0,
        (const __half*)x, (const __half*)w, (const __half*)b, (__half*)y,
        N, Cin, Cout, H, W);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
