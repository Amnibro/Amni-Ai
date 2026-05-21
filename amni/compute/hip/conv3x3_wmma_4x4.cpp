#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
#define TILE_S 4
#define SW 4
#define FW 4
#define TILE_F 16
#define BLK_THREADS (32*SW*FW)
__global__ void k_conv3x3_wmma_4x4_s1p1(
    const _Float16* __restrict__ x,
    const _Float16* __restrict__ w,
    const _Float16* __restrict__ b,
    _Float16* __restrict__ y,
    int N, int Cin, int Cout, int H, int W){
    __shared__ _Float16 in_lds[SW*16*16];
    __shared__ _Float16 wt_lds[FW*16*16];
    int tid=threadIdx.x;
    int lane=tid&31;
    int wave=tid>>5;
    int sp_w=wave&(SW-1);
    int fp_w=(wave>>2)&(FW-1);
    int sp_id=lane%16;
    int nts_w_pair=W/(SW*TILE_S);
    int ts=blockIdx.x;
    int tf=blockIdx.y;
    int n=blockIdx.z;
    int spt_h=ts/nts_w_pair;
    int spt_w_pair=ts%nts_w_pair;
    int base_h=spt_h*TILE_S;
    int base_w=spt_w_pair*SW*TILE_S;
    int base_f=tf*FW*TILE_F;
    int my_filter=base_f+fp_w*TILE_F+sp_id;
    v16h C={0};
    int K=Cin*9;
    #define IN_TOTAL (SW*256)
    #define WT_TOTAL (FW*256)
    for(int kc=0;kc<K;kc+=16){
        #pragma unroll
        for(int chunk=0;chunk<(IN_TOTAL+BLK_THREADS-1)/BLK_THREADS;chunk++){
            int idx=chunk*BLK_THREADS+tid;
            if(idx<IN_TOTAL){
                int s=idx>>8;
                int p=(idx>>4)&15;
                int k=idx&15;
                int kk=kc+k;
                int c=kk/9;int khw=kk%9;int kh=khw/3;int kw=khw%3;
                int ph=p>>2;int pw=p&3;
                int sh=base_h+ph;
                int sw=base_w+s*TILE_S+pw;
                int h_in=sh+kh-1;int w_in=sw+kw-1;
                _Float16 xv=(_Float16)0;
                if(c<Cin&&h_in>=0&&h_in<H&&w_in>=0&&w_in<W)xv=x[((n*Cin+c)*H+h_in)*W+w_in];
                in_lds[idx]=xv;
            }
        }
        #pragma unroll
        for(int chunk=0;chunk<(WT_TOTAL+BLK_THREADS-1)/BLK_THREADS;chunk++){
            int idx=chunk*BLK_THREADS+tid;
            if(idx<WT_TOTAL){
                int f=idx>>8;
                int fi=(idx>>4)&15;
                int k=idx&15;
                int kk=kc+k;
                int c=kk/9;int khw=kk%9;int kh=khw/3;int kw=khw%3;
                int my_f=base_f+f*TILE_F+fi;
                _Float16 wv=(_Float16)0;
                if(c<Cin&&my_f<Cout)wv=w[((my_f*Cin+c)*3+kh)*3+kw];
                wt_lds[idx]=wv;
            }
        }
        __syncthreads();
        v16h A,Bv;
        #pragma unroll
        for(int k=0;k<16;k++)A[k]=in_lds[sp_w*256+sp_id*16+k];
        #pragma unroll
        for(int k=0;k<16;k++)Bv[k]=wt_lds[fp_w*256+sp_id*16+k];
        C=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A,Bv,C,false);
        __syncthreads();
    }
    if(my_filter>=Cout)return;
    _Float16 bv=(b!=nullptr)?b[my_filter]:(_Float16)0;
    int my_base_w=base_w+sp_w*TILE_S;
    for(int i=0;i<8;i++){
        int row_id=2*i+(lane/16);
        int oh=base_h+row_id/TILE_S;
        int ow=my_base_w+row_id%TILE_S;
        if(oh<H&&ow<W)y[((n*Cout+my_filter)*H+oh)*W+ow]=(_Float16)((float)C[i*2]+(float)bv);
    }
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int conv3x3_wmma_4x4_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_4x4_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_4x4_run(
    const void* x, const void* w, const void* b, void* y,
    int N, int Cin, int Cout, int H, int W){
    if(W%(SW*TILE_S)!=0||H%TILE_S!=0)return -2;
    int n_spt=(H/TILE_S)*(W/(SW*TILE_S));
    int n_ftt=(Cout+FW*TILE_F-1)/(FW*TILE_F);
    dim3 grid(n_spt, n_ftt, N);
    dim3 block(BLK_THREADS);
    hipLaunchKernelGGL(k_conv3x3_wmma_4x4_s1p1, grid, block, 0, 0,
        (const _Float16*)x, (const _Float16*)w, (const _Float16*)b, (_Float16*)y,
        N, Cin, Cout, H, W);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
