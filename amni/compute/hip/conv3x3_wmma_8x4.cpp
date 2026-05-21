#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
#define TILE_S 4
#define SW 8
#define FW 4
#define SUB_S 2
#define SUB_F 2
#define LDS_S (SW*SUB_S)
#define LDS_F (FW*SUB_F)
#define TILE_F 16
#define BLK_THREADS (32*SW*FW)
__global__ void k_conv3x3_wmma_8x4_s1p1(
    const _Float16* __restrict__ x,
    const _Float16* __restrict__ w,
    const _Float16* __restrict__ b,
    _Float16* __restrict__ y,
    int N, int Cin, int Cout, int H, int W){
    __shared__ __attribute__((aligned(16))) _Float16 in_lds[2][LDS_S*256];
    __shared__ __attribute__((aligned(16))) _Float16 wt_lds[2][LDS_F*256];
    int tid=threadIdx.x;
    int lane=tid&31;
    int wave=tid>>5;
    int sp_w=wave&(SW-1);
    int fp_w=(wave>>3)&(FW-1);
    int sp_id=lane%16;
    int nts_w_pair=W/(LDS_S*TILE_S);
    int ts=blockIdx.x;
    int tf=blockIdx.y;
    int n=blockIdx.z;
    int spt_h=ts/nts_w_pair;
    int spt_w_pair=ts%nts_w_pair;
    int base_h=spt_h*TILE_S;
    int base_w=spt_w_pair*LDS_S*TILE_S;
    int base_f=tf*LDS_F*TILE_F;
    v16h C00={0},C01={0},C10={0},C11={0};
    int K=Cin*9;
    int n_iter=K/16;
    #define LOAD_CHUNK(buf, kc_val) {\
        _Pragma("unroll") \
        for(int u=0;u<4;u++){ \
            int idx=tid*4+u; \
            if(idx<LDS_S*256){ \
                int s_=idx>>8;int p_=(idx>>4)&15;int k_=idx&15; \
                int kk=(kc_val)+k_; \
                int c_=kk/9;int khw=kk%9;int kh_=khw/3;int kw_=khw%3; \
                int ph=p_>>2;int pw=p_&3; \
                int sh=base_h+ph; \
                int sw=base_w+s_*TILE_S+pw; \
                int h_in=sh+kh_-1;int w_in=sw+kw_-1; \
                _Float16 xv=(_Float16)0; \
                if(c_<Cin&&h_in>=0&&h_in<H&&w_in>=0&&w_in<W)xv=x[((n*Cin+c_)*H+h_in)*W+w_in]; \
                in_lds[buf][idx]=xv; \
            } \
        } \
        _Pragma("unroll") \
        for(int u=0;u<2;u++){ \
            int idx=tid*2+u; \
            if(idx<LDS_F*256){ \
                int f_=idx>>8;int fi=(idx>>4)&15;int k_=idx&15; \
                int kk=(kc_val)+k_; \
                int c_=kk/9;int khw=kk%9;int kh_=khw/3;int kw_=khw%3; \
                int my_f=base_f+f_*TILE_F+fi; \
                _Float16 wv=(_Float16)0; \
                if(c_<Cin&&my_f<Cout)wv=w[((my_f*Cin+c_)*3+kh_)*3+kw_]; \
                wt_lds[buf][idx]=wv; \
            } \
        } \
    }
    LOAD_CHUNK(0, 0)
    __syncthreads();
    for(int it=0;it<n_iter;it++){
        int curr=it&1;
        int next_buf=1-curr;
        bool has_next=(it+1<n_iter);
        int next_kc=(it+1)*16;
        if(has_next){
            LOAD_CHUNK(next_buf, next_kc)
        }
        const f16x8* in_v0=(const f16x8*)(&in_lds[curr][(sp_w*2+0)*256+sp_id*16]);
        const f16x8* in_v1=(const f16x8*)(&in_lds[curr][(sp_w*2+1)*256+sp_id*16]);
        const f16x8* wt_v0=(const f16x8*)(&wt_lds[curr][(fp_w*2+0)*256+sp_id*16]);
        const f16x8* wt_v1=(const f16x8*)(&wt_lds[curr][(fp_w*2+1)*256+sp_id*16]);
        f16x8 a0l=in_v0[0],a0h=in_v0[1],a1l=in_v1[0],a1h=in_v1[1];
        f16x8 b0l=wt_v0[0],b0h=wt_v0[1],b1l=wt_v1[0],b1h=wt_v1[1];
        v16h A0,A1,B0,B1;
        #pragma unroll
        for(int i=0;i<8;i++){A0[i]=a0l[i];A0[i+8]=a0h[i];A1[i]=a1l[i];A1[i+8]=a1h[i];B0[i]=b0l[i];B0[i+8]=b0h[i];B1[i]=b1l[i];B1[i+8]=b1h[i];}
        C00=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A0,B0,C00,false);
        C01=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A0,B1,C01,false);
        C10=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A1,B0,C10,false);
        C11=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(A1,B1,C11,false);
        __syncthreads();
    }
    #define STORE(C,subS,subF) {\
        int my_filter=base_f+(fp_w*2+(subF))*TILE_F+sp_id;\
        if(my_filter<Cout){\
            _Float16 bv=(b!=nullptr)?b[my_filter]:(_Float16)0;\
            int my_bw=base_w+(sp_w*2+(subS))*TILE_S;\
            for(int i=0;i<8;i++){\
                int row_id=2*i+(lane/16);\
                int oh=base_h+row_id/TILE_S;\
                int ow=my_bw+row_id%TILE_S;\
                if(oh<H&&ow<W)y[((n*Cout+my_filter)*H+oh)*W+ow]=(_Float16)((float)(C)[i*2]+(float)bv);\
            }\
        }\
    }
    STORE(C00,0,0)
    STORE(C01,0,1)
    STORE(C10,1,0)
    STORE(C11,1,1)
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int conv3x3_wmma_8x4_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_8x4_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_8x4_run(
    const void* x, const void* w, const void* b, void* y,
    int N, int Cin, int Cout, int H, int W){
    if(W%(LDS_S*TILE_S)!=0||H%TILE_S!=0)return -2;
    if((Cin*9)%16!=0)return -3;
    int n_spt=(H/TILE_S)*(W/(LDS_S*TILE_S));
    int n_ftt=(Cout+LDS_F*TILE_F-1)/(LDS_F*TILE_F);
    dim3 grid(n_spt, n_ftt, N);
    dim3 block(BLK_THREADS);
    hipLaunchKernelGGL(k_conv3x3_wmma_8x4_s1p1, grid, block, 0, 0,
        (const _Float16*)x, (const _Float16*)w, (const _Float16*)b, (_Float16*)y,
        N, Cin, Cout, H, W);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
