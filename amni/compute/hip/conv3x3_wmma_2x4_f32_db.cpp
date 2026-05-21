#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
#define TILE_S 4
#define SW 4
#define FW 4
#define SUB_S 2
#define SUB_F 4
#define LDS_S (SW*SUB_S)
#define LDS_F (FW*SUB_F)
#define TILE_F 16
#define BLK_THREADS (32*SW*FW)
__global__ void k_conv3x3_wmma_2x4_f32_db_s1p1(
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
    int fp_w=(wave>>2)&(FW-1);
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
    v8f C00={0},C01={0},C02={0},C03={0},C10={0},C11={0},C12={0},C13={0};
    int K=Cin*9;
    int n_iter=K/16;
    #define LOAD_IN(buf, kc_val) \
        _Pragma("unroll") \
        for(int u=0;u<4;u++){ \
            int idx=tid*4+u; \
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
        }
    #define LOAD_WT(buf, kc_val) \
        _Pragma("unroll") \
        for(int u=0;u<8;u++){ \
            int idx=tid*8+u; \
            int f_=idx>>8;int fi=(idx>>4)&15;int k_=idx&15; \
            int kk=(kc_val)+k_; \
            int c_=kk/9;int khw=kk%9;int kh_=khw/3;int kw_=khw%3; \
            int my_f=base_f+f_*TILE_F+fi; \
            _Float16 wv=(_Float16)0; \
            if(c_<Cin&&my_f<Cout)wv=w[((my_f*Cin+c_)*3+kh_)*3+kw_]; \
            wt_lds[buf][idx]=wv; \
        }
    LOAD_IN(0, 0)
    LOAD_WT(0, 0)
    __syncthreads();
    for(int it=0;it<n_iter;it++){
        int curr=it&1;
        int next_buf=1-curr;
        bool has_next=(it+1<n_iter);
        int next_kc=(it+1)*16;
        if(has_next){
            LOAD_IN(next_buf, next_kc)
            LOAD_WT(next_buf, next_kc)
        }
        const f16x8* in_v0=(const f16x8*)(&in_lds[curr][(sp_w*2+0)*256+sp_id*16]);
        const f16x8* in_v1=(const f16x8*)(&in_lds[curr][(sp_w*2+1)*256+sp_id*16]);
        const f16x8* wt_v0=(const f16x8*)(&wt_lds[curr][(fp_w*4+0)*256+sp_id*16]);
        const f16x8* wt_v1=(const f16x8*)(&wt_lds[curr][(fp_w*4+1)*256+sp_id*16]);
        const f16x8* wt_v2=(const f16x8*)(&wt_lds[curr][(fp_w*4+2)*256+sp_id*16]);
        const f16x8* wt_v3=(const f16x8*)(&wt_lds[curr][(fp_w*4+3)*256+sp_id*16]);
        f16x8 a0l=in_v0[0],a0h=in_v0[1],a1l=in_v1[0],a1h=in_v1[1];
        f16x8 b0l=wt_v0[0],b0h=wt_v0[1],b1l=wt_v1[0],b1h=wt_v1[1];
        f16x8 b2l=wt_v2[0],b2h=wt_v2[1],b3l=wt_v3[0],b3h=wt_v3[1];
        v16h A0,A1,B0,B1,B2,B3;
        #pragma unroll
        for(int i=0;i<8;i++){
            A0[i]=a0l[i];A0[i+8]=a0h[i];A1[i]=a1l[i];A1[i+8]=a1h[i];
            B0[i]=b0l[i];B0[i+8]=b0h[i];B1[i]=b1l[i];B1[i+8]=b1h[i];
            B2[i]=b2l[i];B2[i+8]=b2h[i];B3[i]=b3l[i];B3[i+8]=b3h[i];
        }
        C00=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A0,B0,C00);
        C01=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A0,B1,C01);
        C02=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A0,B2,C02);
        C03=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A0,B3,C03);
        C10=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A1,B0,C10);
        C11=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A1,B1,C11);
        C12=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A1,B2,C12);
        C13=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A1,B3,C13);
        __syncthreads();
    }
    #define STORE_F32(C,subS,subF) {\
        int my_filter=base_f+(fp_w*4+(subF))*TILE_F+(lane%16);\
        if(my_filter<Cout){\
            _Float16 bv=(b!=nullptr)?b[my_filter]:(_Float16)0;\
            int my_bw=base_w+(sp_w*2+(subS))*TILE_S;\
            for(int i=0;i<8;i++){\
                int row=2*i+(lane/16);\
                int oh=base_h+row/TILE_S;\
                int ow=my_bw+row%TILE_S;\
                if(oh<H&&ow<W)y[((n*Cout+my_filter)*H+oh)*W+ow]=(_Float16)((float)(C)[i]+(float)bv);\
            }\
        }\
    }
    STORE_F32(C00,0,0)
    STORE_F32(C01,0,1)
    STORE_F32(C02,0,2)
    STORE_F32(C03,0,3)
    STORE_F32(C10,1,0)
    STORE_F32(C11,1,1)
    STORE_F32(C12,1,2)
    STORE_F32(C13,1,3)
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int conv3x3_wmma_2x4_f32_db_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_2x4_f32_db_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int conv3x3_wmma_2x4_f32_db_run(
    const void* x, const void* w, const void* b, void* y,
    int N, int Cin, int Cout, int H, int W){
    if(W%(LDS_S*TILE_S)!=0||H%TILE_S!=0)return -2;
    if((Cin*9)%16!=0)return -3;
    int n_spt=(H/TILE_S)*(W/(LDS_S*TILE_S));
    int n_ftt=(Cout+LDS_F*TILE_F-1)/(LDS_F*TILE_F);
    dim3 grid(n_spt, n_ftt, N);
    dim3 block(BLK_THREADS);
    hipLaunchKernelGGL(k_conv3x3_wmma_2x4_f32_db_s1p1, grid, block, 0, 0,
        (const _Float16*)x, (const _Float16*)w, (const _Float16*)b, (_Float16*)y,
        N, Cin, Cout, H, W);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
