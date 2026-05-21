#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
#define TILE_S 4
#define TILE_F 16
#define WAVES_PER_BLK 4
#define BLK_F (TILE_F*WAVES_PER_BLK)
__global__ void k_conv3x3_wmma_lds(
    const _Float16* __restrict__ x,
    const _Float16* __restrict__ w,
    const _Float16* __restrict__ b,
    _Float16* __restrict__ y,
    int N, int Cin, int Cout, int H, int W){
    __shared__ _Float16 lds_in[16*16];
    int tid=threadIdx.x;
    int lane=tid&31;
    int wave=tid>>5;
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
    int my_filter=tf*BLK_F+wave*TILE_F+sp_id;
    v16h C={0};
    int K=Cin*9;
    for(int kc=0;kc<K;kc+=16){
        #pragma unroll
        for(int chunk=0;chunk<2;chunk++){
            int idx=chunk*(32*WAVES_PER_BLK)+tid;
            if(idx<256){
                int sp=idx/16;
                int k=idx%16;
                int kk=kc+k;
                int c=kk/9;
                int khw=kk%9;
                int kh=khw/3;
                int kw=khw%3;
                int sh=th_pos*TILE_S+sp/TILE_S;
                int sw=tw_pos*TILE_S+sp%TILE_S;
                int h_in=sh+kh-1;
                int w_in=sw+kw-1;
                _Float16 xv=(_Float16)0;
                if(c<Cin&&h_in>=0&&h_in<H&&w_in>=0&&w_in<W){
                    xv=x[((n*Cin+c)*H+h_in)*W+w_in];
                }
                lds_in[idx]=xv;
            }
        }
        __syncthreads();
        v16h A, Bv;
        #pragma unroll
        for(int k=0;k<16;k++)A[k]=lds_in[sp_id*16+k];
        #pragma unroll
        for(int k=0;k<16;k++){
            int kk=kc+k;
            int c=kk/9;
            int khw=kk%9;
            int kh=khw/3;
            int kw=khw%3;
            _Float16 wv=(_Float16)0;
            if(c<Cin&&my_filter<Cout){
                wv=w[((my_filter*Cin+c)*3+kh)*3+kw];
            }
            Bv[k]=wv;
        }
        C=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A, Bv, C, false);
        __syncthreads();
    }
    if(my_filter>=Cout)return;
    _Float16 bias_v=(b!=nullptr)?b[my_filter]:(_Float16)0;
    for(int i=0;i<8;i++){
        int row_id=2*i+(lane/16);
        int oh=th_pos*TILE_S+row_id/TILE_S;
        int ow=tw_pos*TILE_S+row_id%TILE_S;
        if(oh<H&&ow<W){
            y[((n*Cout+my_filter)*H+oh)*W+ow]=(_Float16)((float)C[i*2]+(float)bias_v);
        }
    }
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int conv3x3_wmma_lds_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_lds_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_lds_run(
    const void* x, const void* w, const void* b, void* y,
    int N, int Cin, int Cout, int H, int W){
    if(W%TILE_S!=0||H%TILE_S!=0)return -2;
    int n_spt=(H/TILE_S)*(W/TILE_S);
    int n_ftt=(Cout+BLK_F-1)/BLK_F;
    dim3 grid(n_spt, n_ftt, N);
    dim3 block(32*WAVES_PER_BLK);
    hipLaunchKernelGGL(k_conv3x3_wmma_lds, grid, block, 0, 0,
        (const _Float16*)x, (const _Float16*)w, (const _Float16*)b, (_Float16*)y,
        N, Cin, Cout, H, W);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
