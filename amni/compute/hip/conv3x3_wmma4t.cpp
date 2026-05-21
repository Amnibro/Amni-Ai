#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
#define TILE_S 4
#define BLK_F 64
__global__ void k_conv3x3_wmma4t_s1p1(
    const _Float16* __restrict__ x,
    const _Float16* __restrict__ w,
    const _Float16* __restrict__ b,
    _Float16* __restrict__ y,
    int N, int Cin, int Cout, int H, int W){
    int lane=threadIdx.x;
    int ts=blockIdx.x;
    int tf=blockIdx.y;
    int n=blockIdx.z;
    int nts_w=W/TILE_S;
    int th_pos=ts/nts_w;
    int tw_pos=ts%nts_w;
    int sp_id=lane%16;
    int sh_off=sp_id/TILE_S;
    int sw_off=sp_id%TILE_S;
    int sp_h=th_pos*TILE_S+sh_off;
    int sp_w=tw_pos*TILE_S+sw_off;
    int f_base=tf*BLK_F+sp_id;
    int f0=f_base,f1=f_base+16,f2=f_base+32,f3=f_base+48;
    v16h C0={0},C1={0},C2={0},C3={0};
    int K=Cin*9;
    for(int kc=0;kc<K;kc+=16){
        v16h A,B0,B1,B2,B3;
        #pragma unroll
        for(int k=0;k<16;k++){
            int kk=kc+k;int c=kk/9;int khw=kk%9;int kh=khw/3;int kw=khw%3;
            int h_in=sp_h+kh-1;int w_in=sp_w+kw-1;
            _Float16 xv=(_Float16)0;
            _Float16 v0=(_Float16)0,v1=(_Float16)0,v2=(_Float16)0,v3=(_Float16)0;
            if(c<Cin){
                if(h_in>=0&&h_in<H&&w_in>=0&&w_in<W)xv=x[((n*Cin+c)*H+h_in)*W+w_in];
                int wbase=(c*3+kh)*3+kw;
                if(f0<Cout)v0=w[f0*Cin*9+wbase];
                if(f1<Cout)v1=w[f1*Cin*9+wbase];
                if(f2<Cout)v2=w[f2*Cin*9+wbase];
                if(f3<Cout)v3=w[f3*Cin*9+wbase];
            }
            A[k]=xv;B0[k]=v0;B1[k]=v1;B2[k]=v2;B3[k]=v3;
        }
        C0=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A,B0,C0,false);
        C1=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A,B1,C1,false);
        C2=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A,B2,C2,false);
        C3=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A,B3,C3,false);
    }
    #define STORE_TILE(C,my_f) {\
        if((my_f)<Cout){\
            _Float16 bv=(b!=nullptr)?b[my_f]:(_Float16)0;\
            for(int i=0;i<8;i++){\
                int row_id=2*i+(lane/16);\
                int oh=th_pos*TILE_S+row_id/TILE_S;\
                int ow=tw_pos*TILE_S+row_id%TILE_S;\
                if(oh<H&&ow<W)y[((n*Cout+(my_f))*H+oh)*W+ow]=(_Float16)((float)C[i*2]+(float)bv);\
            }\
        }\
    }
    STORE_TILE(C0,f0)
    STORE_TILE(C1,f1)
    STORE_TILE(C2,f2)
    STORE_TILE(C3,f3)
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int conv3x3_wmma4t_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma4t_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma4t_run(
    const void* x, const void* w, const void* b, void* y,
    int N, int Cin, int Cout, int H, int W){
    if(W%TILE_S!=0||H%TILE_S!=0)return -2;
    int n_spt=(H/TILE_S)*(W/TILE_S);
    int n_ftt=(Cout+BLK_F-1)/BLK_F;
    dim3 grid(n_spt, n_ftt, N);
    dim3 block(32);
    hipLaunchKernelGGL(k_conv3x3_wmma4t_s1p1, grid, block, 0, 0,
        (const _Float16*)x, (const _Float16*)w, (const _Float16*)b, (_Float16*)y,
        N, Cin, Cout, H, W);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
