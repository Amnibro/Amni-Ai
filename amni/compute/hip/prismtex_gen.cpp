#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define P 17
#define P2 289
#define P3 4913
#define P4 83521
#define BLK 256
#define TILE 16
#define MAX_TEX 512
#define GBLK 256
#define MODE_RAW4 0
#define MODE_BASE17_4 1
#define MODE_FP16_BASE17 2
#define MODE_DUAL_INDEX 3
#define MODE_DENSE_4X4 4
#define MODE_LUT_ADD 5
#define MODE_LUT_MUL 6
#define MODE_NONCE_REF 7
#define MODE_MULTI_PLANE 8
#define MODE_HIER_4D 9
#define MODE_LUT_FUSION 10
#define MODE_FP16_DUAL 11
#define MODE_4TEX_4BIT 12
#define MODE_NONCE_LUT 13
#define MODE_MICRO_LUT 14
#define MODE_GEN_CONST 15
#define MODE_GEN_LINEAR 16
#define MODE_GEN_POLY 17
#define MODE_GEN_POWER 18
#define MODE_GEN_DELTA 19
#define MODE_GEN_MIXED 20
#define MODE_GEN_CYCLE2 21
#define MODE_GEN_ZERO 22
#define MODE_GEN_CONST_PAIR 23
#define MODE_LUT_CHAIN 25
#define MODE_HIER_CHAIN 26
#define MODE_PROCEDURAL_FP16 28
#define MODE_TERNARY_5TRIT 33
static uint8_t* g_bin_buf=nullptr;
static uint8_t* g_una_buf=nullptr;
static uint8_t* g_pow_buf=nullptr;
static uint8_t* g_wbuf[MAX_TEX];
static int g_wtex_w[MAX_TEX],g_wtex_h[MAX_TEX];
static int g_n_wtex=0;
static bool g_init=false;
static uint8_t* g_data_lut=nullptr;
static int g_data_lut_n=0;
static int g_data_lut_k=0;
static uint8_t* g_chain_tex[4]={nullptr,nullptr,nullptr,nullptr};
static int g_chain_w[4]={0,0,0,0};
static int g_chain_h[4]={0,0,0,0};
static uint8_t* g_sec_tex=nullptr;
static int g_sec_w=0,g_sec_h=0;
__device__ __forceinline__ uchar4 buf4(const uint8_t*b,int w,int x,int y){return((const uchar4*)b)[y*w+x];}
__device__ __forceinline__ uint8_t gf17_add_t(const uint8_t*bt,uint8_t a,uint8_t b){return buf4(bt,P,(int)b,(int)a).x;}
__device__ __forceinline__ uint8_t gf17_mul_t(const uint8_t*bt,uint8_t a,uint8_t b){return buf4(bt,P,(int)b,(int)a).y;}
__device__ __forceinline__ uint8_t gf17_sub_t(const uint8_t*bt,uint8_t a,uint8_t b){return buf4(bt,P,(int)b,(int)a).z;}
__device__ __forceinline__ uint8_t gf17_inv_t(const uint8_t*ut,uint8_t a){return buf4(ut,1,0,(int)a).x;}
__device__ __forceinline__ uint8_t gf17_neg_t(const uint8_t*ut,uint8_t a){return buf4(ut,1,0,(int)a).y;}
__device__ __forceinline__ uint8_t gf17_cube_t(const uint8_t*ut,uint8_t a){return buf4(ut,1,0,(int)a).z;}
__device__ __forceinline__ uint8_t gf17_pow_t(const uint8_t*pt,uint8_t base,uint8_t exp){return buf4(pt,P,(int)exp,(int)base).x;}
__device__ __forceinline__ uint8_t gf17_dsp_t(const uint8_t*bt,uint8_t a,uint8_t b){return buf4(bt,P,(int)b,(int)a).w;}
__device__ __forceinline__ uchar4 wbuf_fetch(const uint8_t*wb,int tw,int flat_px){
    int py=flat_px/tw,px=flat_px%tw;
    return((const uchar4*)wb)[py*tw+px];
}
__device__ __forceinline__ uint8_t ternary5_to_gf17(uint8_t packed,int sub){
    int v=sub==0?packed:sub==1?(packed/3):sub==2?(packed/9):sub==3?(packed/27):(packed/81);
    int c=v%3;
    return c==0?16:(c==1?0:1);
}
__device__ uint8_t decode_weight_inline(
    int mode,const uint8_t*wb,int tw,
    const uint8_t*bt,const uint8_t*pt,
    int weight_idx,int gen_blk){
    if(mode==MODE_TERNARY_5TRIT){
        int pi=weight_idx/20,byte_idx=(weight_idx%20)/5,sub=weight_idx%5;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        uint8_t packed=byte_idx==0?t.x:(byte_idx==1?t.y:(byte_idx==2?t.z:t.w));
        return ternary5_to_gf17(packed,sub);
    }
    if(mode<=MODE_BASE17_4||mode==MODE_MULTI_PLANE||mode==MODE_HIER_4D){
        int px_idx=weight_idx/4,sub=weight_idx%4;
        uchar4 t=wbuf_fetch(wb,tw,px_idx);
        uint8_t ch[4]={t.x,t.y,t.z,t.w};
        return ch[sub];
    }
    if(mode==MODE_GEN_CONST){
        int pi=weight_idx/gen_blk;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        return t.x;
    }
    if(mode==MODE_GEN_LINEAR){
        int pi=weight_idx/gen_blk,li=weight_idx%gen_blk;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        uint8_t w0=t.x,delta=t.y;
        uint8_t step=(uint8_t)(li%P);
        return gf17_add_t(bt,w0,gf17_mul_t(bt,delta,step));
    }
    if(mode==MODE_GEN_POLY){
        int pi=weight_idx/gen_blk,li=weight_idx%gen_blk;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        uint8_t a=t.x,b=t.y,c=t.z;
        uint8_t i_mod=(uint8_t)(li%P);
        uint8_t i_sq=gf17_mul_t(bt,i_mod,i_mod);
        return gf17_add_t(bt,gf17_add_t(bt,gf17_mul_t(bt,a,i_sq),gf17_mul_t(bt,b,i_mod)),c);
    }
    if(mode==MODE_GEN_POWER){
        int pi=weight_idx/gen_blk,li=weight_idx%gen_blk;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        uint8_t base=t.x,ratio=t.y;
        uint8_t exp=(uint8_t)(li%P);
        return gf17_mul_t(bt,base,gf17_pow_t(pt,ratio,exp));
    }
    if(mode==MODE_GEN_CYCLE2){
        int pi=weight_idx/gen_blk;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        return(weight_idx%2==0)?t.x:t.y;
    }
    if(mode==MODE_GEN_ZERO)return 0;
    if(mode==MODE_LUT_CHAIN){
        if(!g_data_lut||g_data_lut_k==0)return 0;
        int pi=weight_idx/g_data_lut_k;
        int sub=weight_idx%g_data_lut_k;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        uint32_t idx=(uint32_t)t.x+(uint32_t)t.y*P+(uint32_t)t.z*P2+(uint32_t)t.w*P3;
        int lut_off=(int)idx*g_data_lut_k+sub;
        return g_data_lut[lut_off];
    }
    if(mode==MODE_HIER_CHAIN){
        if(!g_data_lut||g_data_lut_k==0)return 0;
        int pi=weight_idx/g_data_lut_k;
        int sub=weight_idx%g_data_lut_k;
        uchar4 tp=wbuf_fetch(wb,tw,pi);
        uint32_t pri_i=(uint32_t)tp.x+(uint32_t)tp.y*P+(uint32_t)tp.z*P2+(uint32_t)tp.w*P3;
        uint32_t sec_i=0;
        if(g_sec_tex){
            uchar4 ts=buf4(g_sec_tex,g_sec_w,pi%g_sec_w,pi/g_sec_w);
            sec_i=(uint32_t)ts.x+(uint32_t)ts.y*4+(uint32_t)ts.z*16+(uint32_t)ts.w*64;
        }
        uint64_t chain_i=0;
        uint64_t cmul=1;
        for(int ti=0;ti<4;ti++){
            if(g_chain_tex[ti]){
                uchar4 tc=buf4(g_chain_tex[ti],g_chain_w[ti],pi%g_chain_w[ti],pi/g_chain_w[ti]);
                chain_i+=(uint64_t)tc.x*cmul;cmul*=4;
                chain_i+=(uint64_t)tc.y*cmul;cmul*=4;
                chain_i+=(uint64_t)tc.z*cmul;cmul*=4;
                chain_i+=(uint64_t)tc.w*cmul;cmul*=4;
            }
        }
        uint64_t idx64=(uint64_t)pri_i*256ULL*(uint64_t)cmul+(uint64_t)sec_i*(uint64_t)cmul+chain_i;
        int lut_off=(int)(idx64%g_data_lut_n)*g_data_lut_k+sub;
        return g_data_lut[lut_off];
    }
    if(mode==MODE_GEN_DELTA){
        int pi=weight_idx/4,sub=weight_idx%4;
        uchar4 t=wbuf_fetch(wb,tw,pi);
        uint8_t ch[4]={t.x,t.y,t.z,t.w};
        if(pi==0)return ch[sub];
        uchar4 t0=wbuf_fetch(wb,tw,0);
        uint8_t acc=t0.w;
        for(int p=1;p<=pi;p++){
            uchar4 dp=wbuf_fetch(wb,tw,p);
            uint8_t dc[4]={dp.x,dp.y,dp.z,dp.w};
            for(int s=0;s<4;s++){
                acc=gf17_add_t(bt,acc,dc[s]);
                if(p==pi&&s==sub)return acc;
            }
        }
        return acc;
    }
    if(mode==MODE_PROCEDURAL_FP16){
        uchar4 t=wbuf_fetch(wb,tw,weight_idx);
        uint8_t gf3=t.x/P;
        uint8_t d0=t.x%P;
        uint8_t d1=t.y;
        uint8_t d2=t.z;
        uint8_t d3=t.w;
        if(gf3==0){
            return d0;
        }else if(gf3==1){
            return d0;
        }else{
            return d0;
        }
    }
    int px_idx=weight_idx/4,sub=weight_idx%4;
    uchar4 t=wbuf_fetch(wb,tw,px_idx);
    uint8_t ch[4]={t.x,t.y,t.z,t.w};
    return ch[sub];
}
__global__ void k_ptex_gen_decode(
    const uint8_t*__restrict__ wb,int tw,
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ ut,const uint8_t*__restrict__ pt,
    int mode,int gen_blk,
    uint8_t*__restrict__ out,int n_weights){
    int wi=blockIdx.x*blockDim.x+threadIdx.x;
    if(wi>=n_weights)return;
    out[wi]=decode_weight_inline(mode,wb,tw,bt,pt,wi,gen_blk);
}
__global__ void k_ptex_gen_matmul(
    const uint8_t*__restrict__ wb,int tw,
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ pt,
    int mode,int gen_blk,
    const uint8_t*__restrict__ A,
    uint8_t*__restrict__ C,
    int M,int K,int N){
    __shared__ uint8_t As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x;
        int wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        Ws[threadIdx.y][threadIdx.x]=(col<N&&wk<K)?decode_weight_inline(mode,wb,tw,bt,pt,wk*N+col,gen_blk):0;
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            uint8_t prod=gf17_mul_t(bt,As[threadIdx.y][k],Ws[k][threadIdx.x]);
            acc=gf17_add_t(bt,acc,prod);
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void k_ptex_gen_matmul_t(
    const uint8_t*__restrict__ wb,int tw,
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ pt,
    int mode,int gen_blk,
    const uint8_t*__restrict__ A,
    uint8_t*__restrict__ C,
    int M,int K,int N){
    __shared__ uint8_t As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x;
        int wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        Ws[threadIdx.y][threadIdx.x]=(col<N&&wk<K)?decode_weight_inline(mode,wb,tw,bt,pt,col*K+wk,gen_blk):0;
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            uint8_t prod=gf17_mul_t(bt,As[threadIdx.y][k],Ws[k][threadIdx.x]);
            acc=gf17_add_t(bt,acc,prod);
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void k_ptex_gen_embed(
    const uint8_t*__restrict__ wb,int tw,
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ pt,
    int mode,int gen_blk,
    const int*__restrict__ ids,
    uint8_t*__restrict__ out,
    int S,int D){
    int si=blockIdx.y,di=blockIdx.x*blockDim.x+threadIdx.x;
    if(si>=S||di>=D)return;
    int tok=ids[si];
    out[si*D+di]=decode_weight_inline(mode,wb,tw,bt,pt,tok*D+di,gen_blk);
}
__global__ void k_ptex_lut_add(
    const uint8_t*__restrict__ bt,
    const uint8_t*__restrict__ a,const uint8_t*__restrict__ b,
    uint8_t*__restrict__ c,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    c[i]=gf17_add_t(bt,a[i],b[i]);
}
__global__ void k_ptex_lut_mul(
    const uint8_t*__restrict__ bt,
    const uint8_t*__restrict__ a,const uint8_t*__restrict__ b,
    uint8_t*__restrict__ c,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    c[i]=gf17_mul_t(bt,a[i],b[i]);
}
__global__ void k_ptex_lut_sub(
    const uint8_t*__restrict__ bt,
    const uint8_t*__restrict__ a,const uint8_t*__restrict__ b,
    uint8_t*__restrict__ c,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    c[i]=gf17_sub_t(bt,a[i],b[i]);
}
__global__ void k_ptex_activate(
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ ut,
    const uint8_t*__restrict__ gate,const uint8_t*__restrict__ up,
    uint8_t*__restrict__ out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint8_t g=gate[i],u=up[i];
    uint8_t g_act=gf17_cube_t(ut,g);
    out[i]=gf17_mul_t(bt,g_act,u);
}
__global__ void k_ptex_rms_norm(
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ ut,
    const uint8_t*__restrict__ x,uint8_t*__restrict__ out,
    int rows,int cols){
    int r=blockIdx.x;
    if(r>=rows)return;
    __shared__ unsigned int sum_sq_u32;
    if(threadIdx.x==0)sum_sq_u32=0;
    __syncthreads();
    for(int c=threadIdx.x;c<cols;c+=blockDim.x){
        uint8_t v=x[r*cols+c];
        uint8_t sq=gf17_mul_t(bt,v,v);
        atomicAdd(&sum_sq_u32,(unsigned int)sq);
    }
    __syncthreads();
    uint8_t sum_mod=(uint8_t)(sum_sq_u32%P);
    uint8_t inv_rms=gf17_inv_t(ut,(sum_mod==0)?1:sum_mod);
    for(int c=threadIdx.x;c<cols;c+=blockDim.x){
        out[r*cols+c]=gf17_mul_t(bt,x[r*cols+c],inv_rms);
    }
}
__global__ void k_ptex_neg_score(
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ ut,
    const uint8_t*__restrict__ Q,const uint8_t*__restrict__ K_,
    uint8_t*__restrict__ out,
    int B,int H,int S,int T,int Hd){
    int bh=blockIdx.z,si=blockIdx.y*blockDim.y+threadIdx.y,ti=blockIdx.x*blockDim.x+threadIdx.x;
    int b=bh/H,h=bh%H;
    if(b>=B||si>=S||ti>=T)return;
    unsigned int acc=0;
    for(int d=0;d<Hd;d++){
        uint8_t q=Q[((b*H+h)*S+si)*Hd+d];
        uint8_t k=K_[((b*H+h)*T+ti)*Hd+d];
        acc+=(unsigned int)gf17_dsp_t(bt,q,k);
    }
    out[((b*H+h)*S+si)*T+ti]=(uint8_t)(acc%P);
}
__global__ void k_ptex_softmax(
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ ut,
    const uint8_t*__restrict__ sc,uint8_t*__restrict__ out,
    int B,int H,int S,int T){
    int bh=blockIdx.y,si=blockIdx.x*blockDim.x+threadIdx.x;
    int b=bh/H,h=bh%H;
    if(b>=B||si>=S)return;
    unsigned int sum=0;
    for(int t=0;t<T;t++)sum+=(unsigned int)sc[((b*H+h)*S+si)*T+t];
    uint8_t sum_mod=(uint8_t)(sum%P);
    uint8_t inv_sum=(sum_mod==0)?1:gf17_inv_t(ut,sum_mod);
    for(int t=0;t<T;t++){
        uint8_t v=sc[((b*H+h)*S+si)*T+t];
        out[((b*H+h)*S+si)*T+t]=gf17_mul_t(bt,v,inv_sum);
    }
}
__global__ void k_ptex_apply_v(
    const uint8_t*__restrict__ bt,
    const uint8_t*__restrict__ attn,const uint8_t*__restrict__ V,
    uint8_t*__restrict__ out,
    int B,int H,int S,int T,int Hd){
    int bh=blockIdx.z,si=blockIdx.y*blockDim.y+threadIdx.y,di=blockIdx.x*blockDim.x+threadIdx.x;
    int b=bh/H,h=bh%H;
    if(b>=B||si>=S||di>=Hd)return;
    uint8_t acc=0;
    for(int t=0;t<T;t++){
        uint8_t a=attn[((b*H+h)*S+si)*T+t];
        uint8_t v=V[((b*H+h)*T+t)*Hd+di];
        acc=gf17_add_t(bt,acc,gf17_mul_t(bt,a,v));
    }
    out[((b*H+h)*S+si)*Hd+di]=acc;
}
__global__ void k_ptex_xpose_bshd(
    const uint8_t*__restrict__ in,uint8_t*__restrict__ out,
    int B,int S,int H,int Hd){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=B*S*H*Hd;
    if(idx>=total)return;
    int b=idx/(S*H*Hd),rem=idx%(S*H*Hd);
    int s=rem/(H*Hd),rem2=rem%(H*Hd);
    int h=rem2/Hd,d=rem2%Hd;
    out[((b*H+h)*S+s)*Hd+d]=in[idx];
}
__global__ void k_ptex_xpose_bhsd(
    const uint8_t*__restrict__ in,uint8_t*__restrict__ out,
    int B,int H,int S,int Hd){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=B*H*S*Hd;
    if(idx>=total)return;
    int b=idx/(H*S*Hd),rem=idx%(H*S*Hd);
    int h=rem/(S*Hd),rem2=rem%(S*Hd);
    int s=rem2/Hd,d=rem2%Hd;
    out[((b*S+s)*H+h)*Hd+d]=in[idx];
}
__global__ void k_ptex_repeat_kv(
    const uint8_t*__restrict__ in,uint8_t*__restrict__ out,
    int B,int Hkv,int H,int T,int Hd){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=B*H*T*Hd;
    if(idx>=total)return;
    int b=idx/(H*T*Hd),rem=idx%(H*T*Hd);
    int h=rem/(T*Hd),rem2=rem%(T*Hd);
    int t=rem2/Hd,d=rem2%Hd;
    int hkv=h/(H/Hkv);
    out[idx]=in[((b*Hkv+hkv)*T+t)*Hd+d];
}
__global__ void k_ptex_dense16_matmul_t(
    const uint8_t*__restrict__ b0,const uint8_t*__restrict__ b1,
    const uint8_t*__restrict__ b2,const uint8_t*__restrict__ b3,
    int tw,const uint8_t*__restrict__ bt,
    const uint8_t*__restrict__ A,
    uint8_t*__restrict__ C,
    int M,int K,int N){
    __shared__ uint8_t As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x;
        int wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        if(col<N&&wk<K){
            int flat=col*K+wk;
            int gi=flat/16,sub=flat%16;
            int ti=sub/4,ci=sub%4;
            int py=gi/tw,px=gi%tw;
            const uint8_t*tx=(ti==0)?b0:(ti==1)?b1:(ti==2)?b2:b3;
            uchar4 texel=((const uchar4*)tx)[py*tw+px];
            uint8_t ch[4]={texel.x,texel.y,texel.z,texel.w};
            Ws[threadIdx.y][threadIdx.x]=ch[ci];
        }else{Ws[threadIdx.y][threadIdx.x]=0;}
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc=gf17_add_t(bt,acc,gf17_mul_t(bt,As[threadIdx.y][k],Ws[k][threadIdx.x]));
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void k_ptex_mixed_matmul_t(
    const uint8_t*__restrict__ wb,int tw,
    const uint8_t*__restrict__ mb,int mtw,
    const uint8_t*__restrict__ bt,const uint8_t*__restrict__ pt,
    int gen_blk,
    const uint8_t*__restrict__ A,
    uint8_t*__restrict__ C,
    int M,int K,int N){
    __shared__ uint8_t As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x;
        int wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        if(col<N&&wk<K){
            int flat=col*K+wk;
            int blk_idx=flat/gen_blk;
            int mpy=blk_idx/mtw,mpx=blk_idx%mtw;
            uchar4 mt=((const uchar4*)mb)[mpy*mtw+mpx];
            int bmode=(int)mt.x;
            Ws[threadIdx.y][threadIdx.x]=decode_weight_inline(bmode,wb,tw,bt,pt,flat,gen_blk);
        }else{Ws[threadIdx.y][threadIdx.x]=0;}
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc=gf17_add_t(bt,acc,gf17_mul_t(bt,As[threadIdx.y][k],Ws[k][threadIdx.x]));
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
static uint8_t* make_buf(const uint8_t*host,int w,int h){
    size_t sz=(size_t)w*h*4;
    uint8_t*d=nullptr;
    hipMalloc(&d,sz);
    hipMemcpy(d,host,sz,hipMemcpyHostToDevice);
    return d;
}
static void free_buf(uint8_t**buf){
    if(*buf){hipFree(*buf);*buf=nullptr;}
}
extern "C"{
int ptex_gen_init(int dev){
    if(g_init)return 0;
    if(hipSetDevice(dev)!=hipSuccess)return-1;
    uint8_t bin[P*P*4];
    memset(bin,0,sizeof(bin));
    for(int a=0;a<P;a++)for(int b=0;b<P;b++){
        int off=(a*P+b)*4;
        bin[off+0]=(uint8_t)((a+b)%P);
        bin[off+1]=(uint8_t)((a*b)%P);
        bin[off+2]=(uint8_t)((a-b+P)%P);
        bin[off+3]=0;
    }
    int8_t alt_enc[P][4];
    memset(alt_enc,0,sizeof(alt_enc));
    for(int code=0;code<16;code++){
        int8_t bits[4];
        for(int i=0;i<4;i++)bits[i]=((code>>(3-i))&1)?1:-1;
        int val=((8*bits[0]+4*bits[1]+2*bits[2]+bits[3])%P+P)%P;
        for(int i=0;i<4;i++)alt_enc[val][i]=bits[i];
    }
    for(int a=0;a<P;a++)for(int b=0;b<P;b++){
        int8_t*ea=alt_enc[a],*eb=alt_enc[b];
        int16_t c3=ea[0]*eb[3]+ea[1]*eb[2]+ea[2]*eb[1]+ea[3]*eb[0];
        int16_t c2=ea[1]*eb[3]+ea[2]*eb[2]+ea[3]*eb[1]-ea[0]*eb[0];
        int16_t c1=ea[2]*eb[3]+ea[3]*eb[2]-ea[0]*eb[1]-ea[1]*eb[0];
        int16_t c0=ea[3]*eb[3]-ea[0]*eb[2]-ea[1]*eb[1]-ea[2]*eb[0];
        int val=((int)(8*c3+4*c2+2*c1+c0)%P+P)%P;
        bin[(a*P+b)*4+3]=(uint8_t)val;
    }
    g_bin_buf=make_buf(bin,P,P);
    uint8_t una[P*4];
    memset(una,0,sizeof(una));
    uint8_t inv[P]={0};
    for(int i=1;i<P;i++)for(int j=1;j<P;j++)if((i*j)%P==1){inv[i]=(uint8_t)j;break;}
    for(int i=0;i<P;i++){
        una[i*4+0]=inv[i];
        una[i*4+1]=(uint8_t)((P-i)%P);
        una[i*4+2]=(uint8_t)((i*i*(uint32_t)i)%P);
        una[i*4+3]=0;
    }
    g_una_buf=make_buf(una,1,P);
    uint8_t pow_t[P*P*4];
    memset(pow_t,0,sizeof(pow_t));
    for(int base=0;base<P;base++){
        uint8_t v=1;
        for(int exp=0;exp<P;exp++){
            pow_t[(base*P+exp)*4+0]=v;
            pow_t[(base*P+exp)*4+1]=0;
            pow_t[(base*P+exp)*4+2]=0;
            pow_t[(base*P+exp)*4+3]=0;
            v=(uint8_t)((v*(uint32_t)base)%P);
        }
    }
    g_pow_buf=make_buf(pow_t,P,P);
    memset(g_wbuf,0,sizeof(g_wbuf));
    g_n_wtex=0;g_init=true;
    return 0;
}
void ptex_gen_shutdown(){
    if(!g_init)return;
    free_buf(&g_bin_buf);
    free_buf(&g_una_buf);
    free_buf(&g_pow_buf);
    if(g_data_lut){hipFree(g_data_lut);g_data_lut=nullptr;}
    g_data_lut_n=0;g_data_lut_k=0;
    for(int i=0;i<g_n_wtex;i++)free_buf(&g_wbuf[i]);
    g_n_wtex=0;g_init=false;
}
int ptex_gen_bind_weight(const void*host_rgba,int w,int h){
    if(g_n_wtex>=MAX_TEX)return-1;
    int idx=g_n_wtex;
    g_wbuf[idx]=make_buf((const uint8_t*)host_rgba,w,h);
    g_wtex_w[idx]=w;g_wtex_h[idx]=h;
    g_n_wtex++;return idx;
}
void ptex_gen_free_weight(int idx){
    if(idx<0||idx>=g_n_wtex)return;
    free_buf(&g_wbuf[idx]);
    g_wtex_w[idx]=0;g_wtex_h[idx]=0;
}
int ptex_gen_get_tex_w(int idx){return(idx>=0&&idx<g_n_wtex)?g_wtex_w[idx]:0;}
int ptex_gen_get_tex_h(int idx){return(idx>=0&&idx<g_n_wtex)?g_wtex_h[idx]:0;}
void*ptex_gen_alloc(size_t n){void*p=nullptr;hipMalloc(&p,n);return p;}
void ptex_gen_free(void*p){if(p)hipFree(p);}
int ptex_gen_h2d(void*d,const void*h,size_t n){return(hipMemcpy(d,h,n,hipMemcpyHostToDevice)==hipSuccess)?0:-1;}
int ptex_gen_d2h(void*h,const void*d,size_t n){return(hipMemcpy(h,d,n,hipMemcpyDeviceToHost)==hipSuccess)?0:-1;}
int ptex_gen_d2d(void*d,const void*s,size_t n){return(hipMemcpy(d,s,n,hipMemcpyDeviceToDevice)==hipSuccess)?0:-1;}
int ptex_gen_sync(){return(hipDeviceSynchronize()==hipSuccess)?0:-1;}
int ptex_gen_decode(int tex_idx,int mode,int gen_blk,void*out,int n){
    if(tex_idx<0||tex_idx>=g_n_wtex)return-1;
    k_ptex_gen_decode<<<(n+BLK-1)/BLK,BLK>>>(
        g_wbuf[tex_idx],g_wtex_w[tex_idx],
        g_bin_buf,g_una_buf,g_pow_buf,
        mode,gen_blk,(uint8_t*)out,n);
    return 0;
}
int ptex_gen_matmul(int tex_idx,int mode,int gen_blk,
    const void*A,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_n_wtex)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_ptex_gen_matmul<<<gr,bl>>>(
        g_wbuf[tex_idx],g_wtex_w[tex_idx],
        g_bin_buf,g_pow_buf,
        mode,gen_blk,
        (const uint8_t*)A,(uint8_t*)C,M,K,N);
    return 0;
}
int ptex_gen_matmul_t(int tex_idx,int mode,int gen_blk,
    const void*A,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_n_wtex)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_ptex_gen_matmul_t<<<gr,bl>>>(
        g_wbuf[tex_idx],g_wtex_w[tex_idx],
        g_bin_buf,g_pow_buf,
        mode,gen_blk,
        (const uint8_t*)A,(uint8_t*)C,M,K,N);
    return 0;
}
int ptex_gen_embed(int tex_idx,int mode,int gen_blk,
    const void*ids,void*out,int S,int D){
    if(tex_idx<0||tex_idx>=g_n_wtex)return-1;
    dim3 bl(BLK),gr((D+BLK-1)/BLK,S);
    k_ptex_gen_embed<<<gr,bl>>>(
        g_wbuf[tex_idx],g_wtex_w[tex_idx],
        g_bin_buf,g_pow_buf,
        mode,gen_blk,
        (const int*)ids,(uint8_t*)out,S,D);
    return 0;
}
int ptex_gen_elem_add(const void*a,const void*b,void*c,int n){
    k_ptex_lut_add<<<(n+BLK-1)/BLK,BLK>>>(g_bin_buf,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
int ptex_gen_elem_mul(const void*a,const void*b,void*c,int n){
    k_ptex_lut_mul<<<(n+BLK-1)/BLK,BLK>>>(g_bin_buf,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
int ptex_gen_elem_sub(const void*a,const void*b,void*c,int n){
    k_ptex_lut_sub<<<(n+BLK-1)/BLK,BLK>>>(g_bin_buf,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
int ptex_gen_activate(const void*gate,const void*up,void*out,int n){
    k_ptex_activate<<<(n+BLK-1)/BLK,BLK>>>(g_bin_buf,g_una_buf,(const uint8_t*)gate,(const uint8_t*)up,(uint8_t*)out,n);return 0;
}
int ptex_gen_rms_norm(const void*x,void*out,int rows,int cols){
    k_ptex_rms_norm<<<rows,min(cols,BLK)>>>(g_bin_buf,g_una_buf,(const uint8_t*)x,(uint8_t*)out,rows,cols);return 0;
}
int ptex_gen_neg_score(const void*Q,const void*K_,void*out,int B,int H,int S,int T,int Hd){
    dim3 bl(TILE,TILE),gr((T+TILE-1)/TILE,(S+TILE-1)/TILE,B*H);
    k_ptex_neg_score<<<gr,bl>>>(g_bin_buf,g_una_buf,(const uint8_t*)Q,(const uint8_t*)K_,(uint8_t*)out,B,H,S,T,Hd);return 0;
}
int ptex_gen_softmax(const void*sc,void*out,int B,int H,int S,int T){
    k_ptex_softmax<<<dim3(S,B*H),1>>>(g_bin_buf,g_una_buf,(const uint8_t*)sc,(uint8_t*)out,B,H,S,T);return 0;
}
int ptex_gen_apply_v(const void*attn,const void*V,void*out,int B,int H,int S,int T,int Hd){
    dim3 bl(TILE,TILE),gr((Hd+TILE-1)/TILE,(S+TILE-1)/TILE,B*H);
    k_ptex_apply_v<<<gr,bl>>>(g_bin_buf,(const uint8_t*)attn,(const uint8_t*)V,(uint8_t*)out,B,H,S,T,Hd);return 0;
}
int ptex_gen_xpose_bshd(const void*in,void*out,int B,int S,int H,int Hd){
    int n=B*S*H*Hd;
    k_ptex_xpose_bshd<<<(n+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,S,H,Hd);return 0;
}
int ptex_gen_xpose_bhsd(const void*in,void*out,int B,int H,int S,int Hd){
    int n=B*H*S*Hd;
    k_ptex_xpose_bhsd<<<(n+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,H,S,Hd);return 0;
}
int ptex_gen_repeat_kv(const void*in,void*out,int B,int Hkv,int H,int T,int Hd){
    int n=B*H*T*Hd;
    k_ptex_repeat_kv<<<(n+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,Hkv,H,T,Hd);return 0;
}
int ptex_gen_dense16_matmul_t(int t0,int t1,int t2,int t3,
    const void*A,void*C,int M,int K,int N){
    if(t0<0||t1<0||t2<0||t3<0)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_ptex_dense16_matmul_t<<<gr,bl>>>(
        g_wbuf[t0],g_wbuf[t1],g_wbuf[t2],g_wbuf[t3],
        g_wtex_w[t0],g_bin_buf,
        (const uint8_t*)A,(uint8_t*)C,M,K,N);
    return 0;
}
int ptex_gen_mixed_matmul_t(int wtex_idx,int mtex_idx,int gen_blk,
    const void*A,void*C,int M,int K,int N){
    if(wtex_idx<0||mtex_idx<0)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_ptex_mixed_matmul_t<<<gr,bl>>>(
        g_wbuf[wtex_idx],g_wtex_w[wtex_idx],
        g_wbuf[mtex_idx],g_wtex_w[mtex_idx],
        g_bin_buf,g_pow_buf,gen_blk,
        (const uint8_t*)A,(uint8_t*)C,M,K,N);
    return 0;
}
int ptex_gen_bind_data_lut(const void*host_data,int n_entries,int block_size){
    if(g_data_lut){hipFree(g_data_lut);g_data_lut=nullptr;}
    size_t sz=(size_t)n_entries*block_size;
    if(hipMalloc(&g_data_lut,sz)!=hipSuccess)return-1;
    if(hipMemcpy(g_data_lut,host_data,sz,hipMemcpyHostToDevice)!=hipSuccess){hipFree(g_data_lut);g_data_lut=nullptr;return-1;}
    g_data_lut_n=n_entries;g_data_lut_k=block_size;
    return 0;
}
void ptex_gen_free_data_lut(){
    if(g_data_lut){hipFree(g_data_lut);g_data_lut=nullptr;}
    g_data_lut_n=0;g_data_lut_k=0;
}
int ptex_gen_bind_chain_tex(int idx,const void*host_data,int w,int h){
    if(idx<0||idx>=4)return-1;
    if(g_chain_tex[idx]){hipFree(g_chain_tex[idx]);g_chain_tex[idx]=nullptr;}
    size_t sz=(size_t)w*h*4;
    if(hipMalloc(&g_chain_tex[idx],sz)!=hipSuccess)return-1;
    if(hipMemcpy(g_chain_tex[idx],host_data,sz,hipMemcpyHostToDevice)!=hipSuccess){hipFree(g_chain_tex[idx]);g_chain_tex[idx]=nullptr;return-1;}
    g_chain_w[idx]=w;g_chain_h[idx]=h;
    return 0;
}
void ptex_gen_free_chain_tex(int idx){
    if(idx<0||idx>=4)return;
    if(g_chain_tex[idx]){hipFree(g_chain_tex[idx]);g_chain_tex[idx]=nullptr;}
    g_chain_w[idx]=0;g_chain_h[idx]=0;
}
int ptex_gen_bind_sec_tex(const void*host_data,int w,int h){
    if(g_sec_tex){hipFree(g_sec_tex);g_sec_tex=nullptr;}
    size_t sz=(size_t)w*h*4;
    if(hipMalloc(&g_sec_tex,sz)!=hipSuccess)return-1;
    if(hipMemcpy(g_sec_tex,host_data,sz,hipMemcpyHostToDevice)!=hipSuccess){hipFree(g_sec_tex);g_sec_tex=nullptr;return-1;}
    g_sec_w=w;g_sec_h=h;
    return 0;
}
void ptex_gen_free_sec_tex(){
    if(g_sec_tex){hipFree(g_sec_tex);g_sec_tex=nullptr;}
    g_sec_w=0;g_sec_h=0;
}
void ptex_gen_free_all_chain(){
    for(int i=0;i<4;i++)ptex_gen_free_chain_tex(i);
    ptex_gen_free_sec_tex();
    ptex_gen_free_data_lut();
}
}
