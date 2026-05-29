#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#define P 17
#define P2 (P*P)
#define TILE 16
#define BLK 256
#define VQ_SZ 256
#define VQ_DIM 4
#define ATEX_MAGIC 0x41544558
#define ATEX_VER 1
#define ATEX_F_RGBA16 1
#define ATEX_F_TERNARY 2
#define TEX_FMT_RGBA8 0
#define TEX_FMT_RGBA16 1
static uint8_t *g_mul=nullptr,*g_add=nullptr,*g_score=nullptr;
static uint8_t *g_inv=nullptr,*g_cube=nullptr;
static bool g_ok=false;
static int g_disp_cap=0;
static uint64_t *g_disp_gu_h=nullptr,*g_disp_dn_h=nullptr;
static int32_t *g_disp_gu_tw=nullptr,*g_disp_dn_tw=nullptr,*g_disp_gu_fmt=nullptr,*g_disp_dn_fmt=nullptr;
__constant__ uint8_t _LEG_DEC[16];
struct TexHandle {
    hipTextureObject_t tex;
    hipArray_t arr;
    int w,h;
    int fmt;
};
#define MAX_TEX 32768
static TexHandle g_tex[MAX_TEX];
static int g_ntex=0;
static uint8_t *g_vq_cb=nullptr;
static void compute_tables(uint8_t*mul,uint8_t*add,uint8_t*sc,uint8_t*inv,uint8_t*cube){
    for(int i=0;i<P;i++){
        for(int j=0;j<P;j++){mul[i*P+j]=(uint8_t)((i*j)%P);add[i*P+j]=(uint8_t)((i+j)%P);}
        cube[i]=(uint8_t)((i*i%P)*i%P);inv[i]=0;
    }
    for(int i=1;i<P;i++){int r=1,b=i,e=P-2;while(e>0){if(e&1)r=r*b%P;b=b*b%P;e>>=1;}inv[i]=(uint8_t)r;}
    int8_t ae[P][4]={};
    for(int c=0;c<16;c++){
        int8_t bits[4];for(int b=0;b<4;b++)bits[b]=((c>>(3-b))&1)?1:-1;
        int v=((8*bits[0]+4*bits[1]+2*bits[2]+bits[3])%P+P)%P;
        for(int b=0;b<4;b++)ae[v][b]=bits[b];
    }
    for(int q=0;q<P;q++)for(int k=0;k<P;k++){
        if(q==0||k==0){sc[q*P+k]=0;continue;}
        int r0=ae[q][0],r1=ae[q][1],r2=ae[q][2],r3=ae[q][3];
        int p0=ae[k][0],p1=ae[k][1],p2=ae[k][2],p3=ae[k][3];
        int c3=r0*p3+r1*p2+r2*p1+r3*p0,c2=r1*p3+r2*p2+r3*p1-r0*p0;
        int c1=r2*p3+r3*p2-r0*p1-r1*p0,c0=r3*p3-r0*p2-r1*p1-r2*p0;
        sc[q*P+k]=(uint8_t)(((8*c3+4*c2+2*c1+c0)%P+P)%P);
    }
}
__device__ __forceinline__ void ld_lut(uint8_t*__restrict__ d,const uint8_t*__restrict__ s,int n,int t,int nth){
    for(int i=t;i<n;i+=nth)d[i]=s[i];
}
__global__ void k_tex_matmul_t(
    const uint8_t*__restrict__ ml,const uint8_t*__restrict__ al,
    const uint8_t*__restrict__ A,
    hipTextureObject_t W_tex,int tex_w,
    uint8_t*__restrict__ C,int M,int K,int N){
    __shared__ uint8_t sm[P2],sa[P2],As[TILE][TILE],Ws[TILE][TILE];
    int tid=threadIdx.y*TILE+threadIdx.x;
    ld_lut(sm,ml,P2,tid,TILE*TILE);ld_lut(sa,al,P2,tid,TILE*TILE);
    __syncthreads();
    int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x,wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        if(col<N&&wk<K){
            int flat_idx=col*K+wk;
            int py=flat_idx/(tex_w*4);
            int px_chan=flat_idx%(tex_w*4);
            int px=px_chan/4;
            int ch=px_chan%4;
            uchar4 texel=tex2D<uchar4>(W_tex,(float)px+0.5f,(float)py+0.5f);
            uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
            Ws[threadIdx.y][threadIdx.x]=vals[ch];
        } else {
            Ws[threadIdx.y][threadIdx.x]=0;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc=sa[acc*P+sm[As[threadIdx.y][k]*P+Ws[k][threadIdx.x]]];
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void k_tex_matmul_vq(
    const uint8_t*__restrict__ ml,const uint8_t*__restrict__ al,
    const uint8_t*__restrict__ A,
    hipTextureObject_t idx_tex,int tex_w,
    const uint8_t*__restrict__ codebook,
    uint8_t*__restrict__ C,int M,int K,int N){
    __shared__ uint8_t sm[P2],sa[P2],cb[VQ_SZ*VQ_DIM],As[TILE][TILE],Ws[TILE][TILE];
    int tid=threadIdx.y*TILE+threadIdx.x;
    ld_lut(sm,ml,P2,tid,TILE*TILE);ld_lut(sa,al,P2,tid,TILE*TILE);
    for(int i=tid;i<VQ_SZ*VQ_DIM;i+=TILE*TILE)cb[i]=codebook[i];
    __syncthreads();
    int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x,wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        if(col<N&&wk<K){
            int w_idx=col*K+wk;
            int blk_idx=w_idx/VQ_DIM;
            int blk_off=w_idx%VQ_DIM;
            int py=blk_idx/(tex_w*4);
            int px_chan=blk_idx%(tex_w*4);
            int px=px_chan/4;
            int ch=px_chan%4;
            uchar4 texel=tex2D<uchar4>(idx_tex,(float)px+0.5f,(float)py+0.5f);
            uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
            uint8_t nonce=vals[ch];
            Ws[threadIdx.y][threadIdx.x]=cb[nonce*VQ_DIM+blk_off];
        } else {
            Ws[threadIdx.y][threadIdx.x]=0;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc=sa[acc*P+sm[As[threadIdx.y][k]*P+Ws[k][threadIdx.x]]];
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void k_tex_embed(
    hipTextureObject_t emb_tex,int tex_w,
    const int32_t*__restrict__ ids,
    uint8_t*__restrict__ out,int S,int D){
    int s=blockIdx.x;if(s>=S)return;
    int tok=ids[s];
    for(int d=threadIdx.x;d<D;d+=blockDim.x){
        int flat=tok*D+d;
        int py=flat/(tex_w*4);
        int px_chan=flat%(tex_w*4);
        int px=px_chan/4;
        int ch=px_chan%4;
        uchar4 texel=tex2D<uchar4>(emb_tex,(float)px+0.5f,(float)py+0.5f);
        uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
        out[s*D+d]=vals[ch];
    }
}
__global__ void k_vq_decode_tex(
    hipTextureObject_t idx_tex,int tex_w,
    const uint8_t*__restrict__ codebook,
    uint8_t*__restrict__ out,int n_blocks){
    __shared__ uint8_t cb[VQ_SZ*VQ_DIM];
    int tid=threadIdx.x;
    for(int i=tid;i<VQ_SZ*VQ_DIM;i+=blockDim.x)cb[i]=codebook[i];
    __syncthreads();
    int bi=blockIdx.x*blockDim.x+tid;
    if(bi>=n_blocks)return;
    int py=bi/(tex_w*4);
    int px_chan=bi%(tex_w*4);
    int px=px_chan/4;
    int ch=px_chan%4;
    uchar4 texel=tex2D<uchar4>(idx_tex,(float)px+0.5f,(float)py+0.5f);
    uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
    uint8_t nonce=vals[ch];
    int base=bi*VQ_DIM;
    #pragma unroll
    for(int d=0;d<VQ_DIM;d++)out[base+d]=cb[nonce*VQ_DIM+d];
}
__global__ void k_fetch_rgba16_indices(hipTextureObject_t tex,int tex_w,const int32_t*__restrict__ idx,uint16_t*__restrict__ out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    int pi=idx[i],px=pi%tex_w,py=pi/tex_w,base=i*4;
    ushort4 p=tex2D<ushort4>(tex,(float)px+0.5f,(float)py+0.5f);
    out[base]=p.x;out[base+1]=p.y;out[base+2]=p.z;out[base+3]=p.w;
}
__global__ void k_elem_add(const uint8_t*__restrict__ al,const uint8_t*__restrict__ a,const uint8_t*__restrict__ b,uint8_t*__restrict__ c,int n){
    __shared__ uint8_t sa[P2];
    ld_lut(sa,al,P2,threadIdx.x,blockDim.x);__syncthreads();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)c[i]=sa[a[i]*P+b[i]];
}
__global__ void k_elem_mul(const uint8_t*__restrict__ ml,const uint8_t*__restrict__ a,const uint8_t*__restrict__ b,uint8_t*__restrict__ c,int n){
    __shared__ uint8_t sm[P2];
    ld_lut(sm,ml,P2,threadIdx.x,blockDim.x);__syncthreads();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)c[i]=sm[a[i]*P+b[i]];
}
__global__ void k_activate(const uint8_t*__restrict__ lut,const uint8_t*__restrict__ x,uint8_t*__restrict__ y,int n){
    __shared__ uint8_t sl[P];
    if(threadIdx.x<P)sl[threadIdx.x]=lut[threadIdx.x];__syncthreads();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)y[i]=sl[x[i]];
}
__global__ void k_rms_norm(
    const uint8_t*__restrict__ ml,const uint8_t*__restrict__ al,const uint8_t*__restrict__ inv,
    const uint8_t*__restrict__ x,uint8_t*__restrict__ y,int rows,int cols){
    int row=blockIdx.x;if(row>=rows)return;
    int tid=threadIdx.x,nth=blockDim.x;
    __shared__ uint8_t sm[P2],sa[P2],si[P],ps[BLK];
    ld_lut(sm,ml,P2,tid,nth);ld_lut(sa,al,P2,tid,nth);
    if(tid<P)si[tid]=inv[tid];__syncthreads();
    uint8_t ls=0;
    for(int j=tid;j<cols;j+=nth){uint8_t v=x[row*cols+j];ls=sa[ls*P+sm[v*P+v]];}
    ps[tid]=ls;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)ps[tid]=sa[ps[tid]*P+ps[tid+s]];__syncthreads();}
    uint8_t iv=si[ps[0]==0?1:ps[0]];__syncthreads();
    for(int j=tid;j<cols;j+=nth)y[row*cols+j]=sm[x[row*cols+j]*P+iv];
}
__global__ void k_neg_score(
    const uint8_t*__restrict__ sl,const uint8_t*__restrict__ al,
    const uint8_t*__restrict__ Q,const uint8_t*__restrict__ K,
    uint8_t*__restrict__ sc,int B,int H,int S,int T,int Hd){
    __shared__ uint8_t ss[P2],sa[P2];
    ld_lut(ss,sl,P2,threadIdx.x,blockDim.x);ld_lut(sa,al,P2,threadIdx.x,blockDim.x);__syncthreads();
    int idx=blockIdx.x*blockDim.x+threadIdx.x,tot=B*H*S*T;
    if(idx>=tot)return;
    int t=idx%T,s=(idx/T)%S,h=(idx/(T*S))%H,b=idx/(T*S*H);
    int qb=((b*H+h)*S+s)*Hd,kb=((b*H+h)*T+t)*Hd;
    uint8_t acc=0;
    for(int d=0;d<Hd;d++)acc=sa[acc*P+ss[Q[qb+d]*P+K[kb+d]]];
    sc[idx]=acc;
}
__global__ void k_attn_norm(
    const uint8_t*__restrict__ ml,const uint8_t*__restrict__ al,const uint8_t*__restrict__ inv,
    uint8_t*__restrict__ sc,int B,int H,int S,int T){
    __shared__ uint8_t sm[P2],sa[P2],si[P];
    ld_lut(sm,ml,P2,threadIdx.x,blockDim.x);ld_lut(sa,al,P2,threadIdx.x,blockDim.x);
    if(threadIdx.x<P)si[threadIdx.x]=inv[threadIdx.x];__syncthreads();
    int idx=blockIdx.x*blockDim.x+threadIdx.x,tot=B*H*S;
    if(idx>=tot)return;
    int base=idx*T;uint8_t ws=0;
    for(int t=0;t<T;t++)ws=sa[ws*P+sc[base+t]];
    uint8_t iw=si[ws==0?1:ws];
    for(int t=0;t<T;t++)sc[base+t]=sm[sc[base+t]*P+iw];
}
__global__ void k_apply_v(
    const uint8_t*__restrict__ ml,const uint8_t*__restrict__ al,
    const uint8_t*__restrict__ sc,const uint8_t*__restrict__ V,
    uint8_t*__restrict__ out,int B,int H,int S,int T,int Hd){
    __shared__ uint8_t sm[P2],sa[P2];
    ld_lut(sm,ml,P2,threadIdx.x,blockDim.x);ld_lut(sa,al,P2,threadIdx.x,blockDim.x);__syncthreads();
    int idx=blockIdx.x*blockDim.x+threadIdx.x,tot=B*H*S*Hd;
    if(idx>=tot)return;
    int d=idx%Hd,s=(idx/Hd)%S,h=(idx/(Hd*S))%H,b=idx/(Hd*S*H);
    int sb=((b*H+h)*S+s)*T,vb=(b*H+h)*T*Hd;
    uint8_t acc=0;
    for(int t=0;t<T;t++)acc=sa[acc*P+sm[sc[sb+t]*P+V[vb+t*Hd+d]]];
    out[((b*H+h)*S+s)*Hd+d]=acc;
}
__global__ void k_xpose_bshd_bhsd(const uint8_t*__restrict__ in,uint8_t*__restrict__ out,int B,int S,int H,int Hd){
    int idx=blockIdx.x*blockDim.x+threadIdx.x,tot=B*S*H*Hd;
    if(idx>=tot)return;
    int hd=idx%Hd,h=(idx/Hd)%H,s=(idx/(Hd*H))%S,b=idx/(Hd*H*S);
    out[((b*H+h)*S+s)*Hd+hd]=in[idx];
}
__global__ void k_xpose_bhsd_bshd(const uint8_t*__restrict__ in,uint8_t*__restrict__ out,int B,int H,int S,int Hd){
    int idx=blockIdx.x*blockDim.x+threadIdx.x,tot=B*H*S*Hd;
    if(idx>=tot)return;
    int hd=idx%Hd,s=(idx/Hd)%S,h=(idx/(Hd*S))%H,b=idx/(Hd*S*H);
    out[((b*S+s)*H+h)*Hd+hd]=in[idx];
}
__global__ void k_repeat_kv(const uint8_t*__restrict__ in,uint8_t*__restrict__ out,int B,int Hkv,int H,int T,int Hd){
    int idx=blockIdx.x*blockDim.x+threadIdx.x,tot=B*H*T*Hd;
    if(idx>=tot)return;
    int hd=idx%Hd,t=(idx/Hd)%T,h=(idx/(Hd*T))%H,b=idx/(Hd*T*H);
    out[idx]=in[((b*Hkv+h/(H/Hkv))*T+t)*Hd+hd];
}
__device__ __forceinline__ float bf16b_to_f32(uint16_t b){uint32_t u=((uint32_t)b)<<16;return __uint_as_float(u);}
__device__ __forceinline__ uint16_t f32_to_bf16b(float v){uint32_t u=__float_as_uint(v);return (uint16_t)((u+0x7FFF+((u>>16)&1))>>16);}
__device__ __forceinline__ float fp16d_from_rgba(uchar4 p){uint16_t bits=(uint16_t)p.x+(uint16_t)p.y*17u+(uint16_t)p.z*289u+(uint16_t)p.w*4913u;__half_raw r;r.x=bits;return __half2float(__half(r));}
__global__ void k_fp16_tex_matmul_t(const uint16_t*__restrict__ A,hipTextureObject_t W_tex,int tex_w,uint16_t*__restrict__ C,int M,int K,int N){
    __shared__ float As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x;
    float acc=0.f;
    int nt=(K+TILE-1)/TILE;
    for(int t=0;t<nt;t++){
        int ak=t*TILE+threadIdx.x,wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?bf16b_to_f32(A[row*K+ak]):0.f;
        if(col<N&&wk<K){int fi=col*K+wk,px=fi%tex_w,py=fi/tex_w;Ws[threadIdx.y][threadIdx.x]=fp16d_from_rgba(tex2D<uchar4>(W_tex,px,py));}
        else Ws[threadIdx.y][threadIdx.x]=0.f;
        __syncthreads();
        for(int k=0;k<TILE;k++)acc+=As[threadIdx.y][k]*Ws[k][threadIdx.x];
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=f32_to_bf16b(acc);
}
__global__ void k_rg_gemv_fused(const uint16_t*__restrict__ x,hipTextureObject_t r_tex,int r_tw,hipTextureObject_t g_tex,int g_tw,hipTextureObject_t b_tex,int b_tw,const float*__restrict__ r_lut,const float*__restrict__ g_lut,const float*__restrict__ b_lut,int routing_k,uint16_t*__restrict__ y,int N,int K){
    __shared__ float sr[4],sg[17],sb[256],red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    if(tid<4)sr[tid]=(tid<routing_k)?r_lut[tid]:0.f;
    if(tid<17)sg[tid]=g_lut[tid];
    sb[tid]=b_lut[tid];
    __syncthreads();
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k;
        int gb=fi>>1,gc=fi&1,gpix=gb>>2,gch=gb&3;
        int bpix=fi>>2,bch=fi&3;
        uchar4 gp=tex2D<uchar4>(g_tex,gpix%g_tw,gpix/g_tw);
        uchar4 bp=tex2D<uchar4>(b_tex,bpix%b_tw,bpix/b_tw);
        uint8_t gby=(gch==0)?gp.x:(gch==1)?gp.y:(gch==2)?gp.z:gp.w;
        uint8_t bby=(bch==0)?bp.x:(bch==1)?bp.y:(bch==2)?bp.z:bp.w;
        int ri;
        if(routing_k==2){
            int rb=fi>>3,rs=fi&7,rpix=rb>>2,rch=rb&3;
            uchar4 rp=tex2D<uchar4>(r_tex,rpix%r_tw,rpix/r_tw);
            uint8_t rby=(rch==0)?rp.x:(rch==1)?rp.y:(rch==2)?rp.z:rp.w;
            ri=(rby>>rs)&1;
        }else{
            int rb=fi>>2,rs=fi&3,rpix=rb>>2,rch=rb&3;
            uchar4 rp=tex2D<uchar4>(r_tex,rpix%r_tw,rpix/r_tw);
            uint8_t rby=(rch==0)?rp.x:(rch==1)?rp.y:(rch==2)?rp.z:rp.w;
            ri=(rby>>(2*rs))&3;
        }
        int gn=(gc==0)?(gby>>4):(gby&0xF);
        float w=sr[ri]+sg[_LEG_DEC[gn]]+sb[bby];
        acc+=w*bf16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_bf16b(red[0]);
}
__global__ void k_rgba_gemv_fused(const uint16_t*__restrict__ x,hipTextureObject_t f_tex,int f_tw,const float*__restrict__ r_lut,const float*__restrict__ g_lut,const float*__restrict__ b_lut,int routing_k,uint16_t*__restrict__ y,int N,int K){
    __shared__ float sr[4],sg[17],sb[256],red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    if(tid<4)sr[tid]=(tid<routing_k)?r_lut[tid]:0.f;
    if(tid<17)sg[tid]=g_lut[tid];
    sb[tid]=b_lut[tid];
    __syncthreads();
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k;
        int pi=fi>>1,sub=fi&1;
        int px=pi%f_tw,py=pi/f_tw;
        uchar4 p=tex2D<uchar4>(f_tex,px,py);
        int ri=(sub==0)?(p.x&0x3):((p.x>>2)&0x3);
        int gn=(sub==0)?(p.y>>4):(p.y&0xF);
        uint8_t bby=(sub==0)?p.z:p.w;
        float w=sr[ri]+sg[_LEG_DEC[gn]]+sb[bby];
        acc+=w*bf16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_bf16b(red[0]);
}
__global__ void k_rg_gemv_fused_offset(const uint16_t*__restrict__ x,hipTextureObject_t r_tex,int r_tw,hipTextureObject_t g_tex,int g_tw,hipTextureObject_t b_tex,int b_tw,const float*__restrict__ r_lut,const float*__restrict__ g_lut,const float*__restrict__ b_lut,int routing_k,uint16_t*__restrict__ y,int N,int K,int weight_offset){
    __shared__ float sr[4],sg[17],sb[256],red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    if(tid<4)sr[tid]=(tid<routing_k)?r_lut[tid]:0.f;
    if(tid<17)sg[tid]=g_lut[tid];
    sb[tid]=b_lut[tid];
    __syncthreads();
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=weight_offset+n*K+k;
        int gb=fi>>1,gc=fi&1,gpix=gb>>2,gch=gb&3;
        int bpix=fi>>2,bch=fi&3;
        uchar4 gp=tex2D<uchar4>(g_tex,gpix%g_tw,gpix/g_tw);
        uchar4 bp=tex2D<uchar4>(b_tex,bpix%b_tw,bpix/b_tw);
        uint8_t gby=(gch==0)?gp.x:(gch==1)?gp.y:(gch==2)?gp.z:gp.w;
        uint8_t bby=(bch==0)?bp.x:(bch==1)?bp.y:(bch==2)?bp.z:bp.w;
        int ri;
        if(routing_k==2){
            int rb=fi>>3,rs=fi&7,rpix=rb>>2,rch=rb&3;
            uchar4 rp=tex2D<uchar4>(r_tex,rpix%r_tw,rpix/r_tw);
            uint8_t rby=(rch==0)?rp.x:(rch==1)?rp.y:(rch==2)?rp.z:rp.w;
            ri=(rby>>rs)&1;
        }else{
            int rb=fi>>2,rs=fi&3,rpix=rb>>2,rch=rb&3;
            uchar4 rp=tex2D<uchar4>(r_tex,rpix%r_tw,rpix/r_tw);
            uint8_t rby=(rch==0)?rp.x:(rch==1)?rp.y:(rch==2)?rp.z:rp.w;
            ri=(rby>>(2*rs))&3;
        }
        int gn=(gc==0)?(gby>>4):(gby&0xF);
        float w=sr[ri]+sg[_LEG_DEC[gn]]+sb[bby];
        acc+=w*bf16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_bf16b(red[0]);
}
__device__ __forceinline__ float fp16b_to_f32(uint16_t b){__half_raw r;r.x=b;return __half2float(__half(r));}
__device__ __forceinline__ uint16_t f32_to_fp16b(float v){__half h=__float2half(v);__half_raw r=static_cast<__half_raw>(h);return r.x;}
__device__ __forceinline__ float ternary5_to_f32(uint8_t packed,int sub){int v=sub==0?packed:sub==1?(packed/3):sub==2?(packed/9):sub==3?(packed/27):(packed/81);int c=v%3;return c==0?-1.f:(c==1?0.f:1.f);}
__global__ void k_gemv_rgba_fp16(const uint16_t*__restrict__ x,hipTextureObject_t W_tex,int W_tw,uint16_t*__restrict__ y,int N,int K){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k,px=fi%W_tw,py=fi/W_tw;
        float w=fp16d_from_rgba(tex2D<uchar4>(W_tex,px,py));
        acc+=w*fp16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_fp16b(red[0]);
}
__device__ __forceinline__ float bf16d_from_rgba(uchar4 p){uint16_t bits=(uint16_t)p.x+(uint16_t)p.y*17u+(uint16_t)p.z*289u+(uint16_t)p.w*4913u;return bf16b_to_f32(bits);}
__global__ void k_gemv_rgba_bf16(const uint16_t*__restrict__ x,hipTextureObject_t W_tex,int W_tw,uint16_t*__restrict__ y,int N,int K){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k,px=fi%W_tw,py=fi/W_tw;
        float w=bf16d_from_rgba(tex2D<uchar4>(W_tex,px,py));
        acc+=w*bf16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_bf16b(red[0]);
}
__global__ void k_bf16_tex_matmul_t(const uint16_t*__restrict__ A,hipTextureObject_t W_tex,int tex_w,uint16_t*__restrict__ C,int M,int K,int N){
    __shared__ float As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x;
    float acc=0.f;
    int nt=(K+TILE-1)/TILE;
    for(int t=0;t<nt;t++){
        int ak=t*TILE+threadIdx.x,wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?bf16b_to_f32(A[row*K+ak]):0.f;
        if(col<N&&wk<K){int fi=col*K+wk,px=fi%tex_w,py=fi/tex_w;Ws[threadIdx.y][threadIdx.x]=bf16d_from_rgba(tex2D<uchar4>(W_tex,px,py));}
        else Ws[threadIdx.y][threadIdx.x]=0.f;
        __syncthreads();
        for(int k=0;k<TILE;k++)acc+=As[threadIdx.y][k]*Ws[k][threadIdx.x];
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=f32_to_bf16b(acc);
}
__global__ void k_fp8_tex_matmul_t(const uint16_t*__restrict__ A,hipTextureObject_t W_tex,int tex_w,const float*__restrict__ lut,uint16_t*__restrict__ C,int M,int K,int N){
    __shared__ float As[TILE][TILE],Ws[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x;
    float acc=0.f;
    int nt=(K+TILE-1)/TILE;
    for(int t=0;t<nt;t++){
        int ak=t*TILE+threadIdx.x,wk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?bf16b_to_f32(A[row*K+ak]):0.f;
        if(col<N&&wk<K){int fi=col*K+wk,pix=fi>>2,ch=fi&3,px=pix%tex_w,py=pix/tex_w;uchar4 p=tex2D<uchar4>(W_tex,px,py);unsigned char b=ch==0?p.x:ch==1?p.y:ch==2?p.z:p.w;Ws[threadIdx.y][threadIdx.x]=lut[b];}
        else Ws[threadIdx.y][threadIdx.x]=0.f;
        __syncthreads();
        for(int k=0;k<TILE;k++)acc+=As[threadIdx.y][k]*Ws[k][threadIdx.x];
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=f32_to_bf16b(acc);
}
__global__ void k_gemv_ternary_fp16(const uint16_t*__restrict__ x,hipTextureObject_t W_tex,int W_tw,uint16_t*__restrict__ y,int N,int K){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k,pi=fi/20,byte_idx=(fi%20)/5,sub=fi%5,px=pi%W_tw,py=pi/W_tw;
        uchar4 p=tex2D<uchar4>(W_tex,px,py);
        uint8_t packed=byte_idx==0?p.x:(byte_idx==1?p.y:(byte_idx==2?p.z:p.w));
        acc+=ternary5_to_f32(packed,sub)*fp16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_fp16b(red[0]);
}
__global__ void k_gemv_rtier_fp16(const uint16_t*__restrict__ x,hipTextureObject_t W_tex,int W_tw,const uint16_t*__restrict__ scales,int N,int K,int kpad,int tile_h,int tile_w,int n_tiles_k,uint16_t*__restrict__ y){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    if(n>=N)return;
    int tile_row=n/tile_h;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int tile_col=k/tile_w;
        int fi=n*kpad+k,pi=fi/20,byte_idx=(fi%20)/5,sub=fi%5,px=pi%W_tw,py=pi/W_tw;
        uchar4 p=tex2D<uchar4>(W_tex,px,py);
        uint8_t packed=byte_idx==0?p.x:(byte_idx==1?p.y:(byte_idx==2?p.z:p.w));
        float t=ternary5_to_f32(packed,sub);
        float s=fp16b_to_f32(scales[tile_row*n_tiles_k+tile_col]);
        acc+=t*s*fp16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_fp16b(red[0]);
}
__global__ void k_gemv_rgba16_fp16(const uint16_t*__restrict__ x,hipTextureObject_t W_tex,int W_tw,uint16_t*__restrict__ y,int N,int K){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k,pi=fi>>2,lane=fi&3,px=pi%W_tw,py=pi/W_tw;
        ushort4 p=tex2D<ushort4>(W_tex,(float)px+0.5f,(float)py+0.5f);
        uint16_t bits=lane==0?p.x:(lane==1?p.y:(lane==2?p.z:p.w));
        acc+=fp16b_to_f32(bits)*fp16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_fp16b(red[0]);
}
__global__ void k_gemv_rgba16_fp16_tiled(const uint16_t*__restrict__ x,hipTextureObject_t W_tex,int tile_h,int tile_w,uint16_t*__restrict__ y,int N,int K){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x;
    int tile_row=n/tile_h,in_row=n%tile_h,tile_pw=tile_w>>2;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int tile_col=k/tile_w,in_col=k%tile_w,px=tile_col*tile_pw+(in_col>>2),py=tile_row*tile_h+in_row;
        ushort4 p=tex2D<ushort4>(W_tex,(float)px+0.5f,(float)py+0.5f);
        uint16_t bits=(in_col&3)==0?p.x:((in_col&3)==1?p.y:((in_col&3)==2?p.z:p.w));
        acc+=fp16b_to_f32(bits)*fp16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)y[n]=f32_to_fp16b(red[0]);
}
__device__ __forceinline__ float tex_fetch_weight(uint64_t tex_h,int tex_w,int tex_fmt,int idx){
    int pi=idx>>2,lane=idx&3,px=pi%tex_w,py=pi/tex_w;
    if(tex_fmt==TEX_FMT_RGBA16){ushort4 p=tex2D<ushort4>((hipTextureObject_t)tex_h,(float)px+0.5f,(float)py+0.5f);uint16_t b=lane==0?p.x:(lane==1?p.y:(lane==2?p.z:p.w));return fp16b_to_f32(b);}
    return fp16d_from_rgba(tex2D<uchar4>((hipTextureObject_t)tex_h,(float)px+0.5f,(float)py+0.5f));
}
__device__ __forceinline__ void tex_fetch_quad(uint64_t tex_h,int tex_w,int tex_fmt,int base_idx,float*out4){
    int pi=base_idx>>2,px=pi%tex_w,py=pi/tex_w;
    if(tex_fmt==TEX_FMT_RGBA16){ushort4 p=tex2D<ushort4>((hipTextureObject_t)tex_h,(float)px+0.5f,(float)py+0.5f);out4[0]=fp16b_to_f32(p.x);out4[1]=fp16b_to_f32(p.y);out4[2]=fp16b_to_f32(p.z);out4[3]=fp16b_to_f32(p.w);}
    else{uchar4 p=tex2D<uchar4>((hipTextureObject_t)tex_h,(float)px+0.5f,(float)py+0.5f);float v=fp16d_from_rgba(p);out4[0]=v;out4[1]=v;out4[2]=v;out4[3]=v;}
}
__global__ void k_dispatch_fused_routes(
    const uint16_t*__restrict__ x,
    const int32_t*__restrict__ token_ids,const float*__restrict__ weights,
    const uint64_t*__restrict__ gu_tex,const int32_t*__restrict__ gu_tw,const int32_t*__restrict__ gu_fmt,const int32_t*__restrict__ gu_off,
    const uint64_t*__restrict__ dn_tex,const int32_t*__restrict__ dn_tw,const int32_t*__restrict__ dn_fmt,const int32_t*__restrict__ dn_off,
    float*__restrict__ y,int num_tokens,int hidden,int N_gu,int K_gu,int N_dn,int K_dn){
    int r=blockIdx.x,tid=threadIdx.x,tok=token_ids[r];
    if(tok<0||tok>=num_tokens)return;
    extern __shared__ float sm[];
    float*gate=sm,*up=sm+K_dn,*xs=sm+2*K_dn;
    int go=gu_off[r],doff=dn_off[r],x_base=tok*hidden;
    uint64_t gu_h=gu_tex[r],dn_h=dn_tex[r];
    int gu_w=gu_tw[r],gu_f=gu_fmt[r],dn_w=dn_tw[r],dn_f=dn_fmt[r];
    int K_gu_q=K_gu&~3;
    for(int k=tid;k<K_gu;k+=blockDim.x)xs[k]=fp16b_to_f32(x[x_base+k]);
    __syncthreads();
    for(int i=tid;i<K_dn;i+=blockDim.x){
        float gs=0.f,us=0.f;
        int g_row=go+i*K_gu,u_row=go+(i+K_dn)*K_gu;
        float gw[4],uw[4];
        int k=0;
        for(;k<K_gu_q;k+=4){
            tex_fetch_quad(gu_h,gu_w,gu_f,g_row+k,gw);
            tex_fetch_quad(gu_h,gu_w,gu_f,u_row+k,uw);
            gs+=xs[k]*gw[0]+xs[k+1]*gw[1]+xs[k+2]*gw[2]+xs[k+3]*gw[3];
            us+=xs[k]*uw[0]+xs[k+1]*uw[1]+xs[k+2]*uw[2]+xs[k+3]*uw[3];
        }
        for(;k<K_gu;k++){gs+=xs[k]*tex_fetch_weight(gu_h,gu_w,gu_f,g_row+k);us+=xs[k]*tex_fetch_weight(gu_h,gu_w,gu_f,u_row+k);}
        float sig=1.0f/(1.0f+expf(-gs));
        gate[i]=gs*sig*us;
    }
    __syncthreads();
    float w=weights[r];
    int K_dn_q=K_dn&~3;
    for(int d=tid;d<N_dn;d+=blockDim.x){
        float acc=0.f;
        int d_row=doff+d*K_dn;
        float dw[4];
        int i=0;
        for(;i<K_dn_q;i+=4){
            tex_fetch_quad(dn_h,dn_w,dn_f,d_row+i,dw);
            acc+=gate[i]*dw[0]+gate[i+1]*dw[1]+gate[i+2]*dw[2]+gate[i+3]*dw[3];
        }
        for(;i<K_dn;i++)acc+=gate[i]*tex_fetch_weight(dn_h,dn_w,dn_f,d_row+i);
        atomicAdd(&y[tok*hidden+d],w*acc);
    }
}
__global__ void k_gemv_rgba16_fp16_batched(const uint16_t*__restrict__ X,hipTextureObject_t W_tex,int W_tw,uint16_t*__restrict__ Y,int M,int N,int K){
    __shared__ float red[BLK];
    int tid=threadIdx.x,n=blockIdx.x,m=blockIdx.y;
    if(m>=M||n>=N)return;
    const uint16_t*x=X+(size_t)m*K;
    float acc=0.f;
    for(int k=tid;k<K;k+=BLK){
        int fi=n*K+k,pi=fi>>2,lane=fi&3,px=pi%W_tw,py=pi/W_tw;
        ushort4 p=tex2D<ushort4>(W_tex,(float)px+0.5f,(float)py+0.5f);
        uint16_t bits=lane==0?p.x:(lane==1?p.y:(lane==2?p.z:p.w));
        acc+=fp16b_to_f32(bits)*fp16b_to_f32(x[k]);
    }
    red[tid]=acc;
    __syncthreads();
    for(int s=BLK/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    if(tid==0)Y[(size_t)m*N+n]=f32_to_fp16b(red[0]);
}
extern "C"{
int ari_init(int dev){
    if(g_ok)return 0;
    if(hipSetDevice(dev)!=hipSuccess)return-1;
    uint8_t hm[P2],ha[P2],hs[P2],hi[P],hc[P];
    compute_tables(hm,ha,hs,hi,hc);
    hipMalloc(&g_mul,P2);hipMemcpy(g_mul,hm,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_add,P2);hipMemcpy(g_add,ha,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_score,P2);hipMemcpy(g_score,hs,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_inv,P);hipMemcpy(g_inv,hi,P,hipMemcpyHostToDevice);
    hipMalloc(&g_cube,P);hipMemcpy(g_cube,hc,P,hipMemcpyHostToDevice);
    uint8_t leg[16]={1,3,9,10,13,5,15,11,16,14,8,7,4,12,2,6};
    hipMemcpyToSymbol(HIP_SYMBOL(_LEG_DEC),leg,16,0,hipMemcpyHostToDevice);
    g_ntex=0;g_ok=true;return 0;
}
void ari_shutdown(){
    if(!g_ok)return;
    for(int i=0;i<g_ntex;i++){
        hipDestroyTextureObject(g_tex[i].tex);
        hipFreeArray(g_tex[i].arr);
    }
    g_ntex=0;
    if(g_vq_cb){hipFree(g_vq_cb);g_vq_cb=nullptr;}
    if(g_disp_gu_h){hipFree(g_disp_gu_h);hipFree(g_disp_dn_h);hipFree(g_disp_gu_tw);hipFree(g_disp_dn_tw);hipFree(g_disp_gu_fmt);hipFree(g_disp_dn_fmt);}
    g_disp_gu_h=g_disp_dn_h=nullptr;g_disp_gu_tw=g_disp_dn_tw=g_disp_gu_fmt=g_disp_dn_fmt=nullptr;g_disp_cap=0;
    hipFree(g_mul);hipFree(g_add);hipFree(g_score);hipFree(g_inv);hipFree(g_cube);
    g_mul=g_add=g_score=g_inv=g_cube=nullptr;g_ok=false;
}
static int g_last_bind_err=0,g_last_bind_stage=0;
extern "C" int ari_get_last_bind_err(){return g_last_bind_err;}
extern "C" int ari_get_last_bind_stage(){return g_last_bind_stage;}
static void bind_log(const char*tag,int w,int h,int err,const char*msg){
    FILE*f=fopen("logs/ari_bind.log","a");
    if(f){fprintf(f,"[ari_bind] %s w=%d h=%d err=%d:%s ntex=%d\n",tag,w,h,err,msg?msg:"",g_ntex);fclose(f);}
    fprintf(stderr,"[ari_bind] %s w=%d h=%d err=%d:%s\n",tag,w,h,err,msg?msg:"");fflush(stderr);
}
static int bind_texture_common(const void*tex_data,int w,int h,int fmt){
    g_last_bind_err=0;g_last_bind_stage=0;
    int slot=-1;
    for(int i=0;i<g_ntex;i++){if(g_tex[i].arr==nullptr){slot=i;break;}}
    if(slot<0){if(g_ntex>=MAX_TEX){g_last_bind_stage=-1;bind_log("MAX_TEX",w,h,0,"");return-1;}slot=g_ntex++;}
    hipChannelFormatDesc desc=fmt==TEX_FMT_RGBA16?hipCreateChannelDesc(16,16,16,16,hipChannelFormatKindUnsigned):hipCreateChannelDesc(8,8,8,8,hipChannelFormatKindUnsigned);
    int pitch=fmt==TEX_FMT_RGBA16?w*8:w*4;
    hipArray_t arr;
    hipError_t e=hipMallocArray(&arr,&desc,w,h);
    if(e!=hipSuccess){g_last_bind_err=(int)e;g_last_bind_stage=1;bind_log("hipMallocArray",w,h,(int)e,hipGetErrorString(e));return-1;}
    e=hipMemcpy2DToArray(arr,0,0,tex_data,pitch,pitch,h,hipMemcpyHostToDevice);
    if(e!=hipSuccess){g_last_bind_err=(int)e;g_last_bind_stage=2;bind_log("hipMemcpy2DToArray",w,h,(int)e,hipGetErrorString(e));hipFreeArray(arr);return-1;}
    hipResourceDesc rd;memset(&rd,0,sizeof(rd));
    rd.resType=hipResourceTypeArray;rd.res.array.array=arr;
    hipTextureDesc td;memset(&td,0,sizeof(td));
    td.addressMode[0]=hipAddressModeClamp;td.addressMode[1]=hipAddressModeClamp;
    td.filterMode=hipFilterModePoint;
    td.readMode=hipReadModeElementType;
    td.normalizedCoords=0;
    hipTextureObject_t texObj=0;
    e=hipCreateTextureObject(&texObj,&rd,&td,nullptr);
    if(e!=hipSuccess){g_last_bind_err=(int)e;g_last_bind_stage=3;bind_log("hipCreateTextureObject",w,h,(int)e,hipGetErrorString(e));hipFreeArray(arr);return-1;}
    g_tex[slot]={texObj,arr,w,h,fmt};
    return slot;
}
int ari_bind_texture(const void*rgba_data,int w,int h){return bind_texture_common(rgba_data,w,h,TEX_FMT_RGBA8);}
int ari_bind_texture_u16(const void*rgba16_data,int w,int h){return bind_texture_common(rgba16_data,w,h,TEX_FMT_RGBA16);}
void ari_free_texture(int idx){
    if(idx<0||idx>=g_ntex||g_tex[idx].arr==nullptr)return;
    hipDestroyTextureObject(g_tex[idx].tex);
    hipFreeArray(g_tex[idx].arr);
    g_tex[idx]={0,nullptr,0,0,0};
}
int ari_tex_update_rect(int idx,int x_px,int y_row,const void*src,int src_pitch_bytes,int width_bytes,int height_rows){
    if(idx<0||idx>=g_ntex||g_tex[idx].arr==nullptr)return-1;
    if(x_px<0||y_row<0||width_bytes<=0||height_rows<=0)return-2;
    if(x_px*4+width_bytes>g_tex[idx].w*4||y_row+height_rows>g_tex[idx].h)return-3;
    hipError_t e=hipMemcpy2DToArray(g_tex[idx].arr,(size_t)(x_px*4),(size_t)y_row,src,(size_t)src_pitch_bytes,(size_t)width_bytes,(size_t)height_rows,hipMemcpyHostToDevice);
    return e==hipSuccess?0:(int)e;
}
int ari_upload_codebook(const void*cb_data,int n_entries,int dim){
    int sz=n_entries*dim;
    if(g_vq_cb)hipFree(g_vq_cb);
    hipMalloc(&g_vq_cb,sz);
    hipMemcpy(g_vq_cb,cb_data,sz,hipMemcpyHostToDevice);
    return 0;
}
void*ari_alloc(size_t n){void*p=nullptr;hipMalloc(&p,n);return p;}
void ari_free(void*p){if(p)hipFree(p);}
int ari_h2d(void*d,const void*s,size_t n){return hipMemcpy(d,s,n,hipMemcpyHostToDevice)==hipSuccess?0:-1;}
int ari_d2h(void*d,const void*s,size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToHost)==hipSuccess?0:-1;}
int ari_d2d(void*d,const void*s,size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToDevice)==hipSuccess?0:-1;}
int ari_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
int ari_tex_matmul_t(const void*A,int tex_idx,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_ntex)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_tex_matmul_t<<<gr,bl>>>(g_mul,g_add,(const uint8_t*)A,g_tex[tex_idx].tex,g_tex[tex_idx].w,(uint8_t*)C,M,K,N);
    return 0;
}
int ari_tex_matmul_vq(const void*A,int tex_idx,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_ntex||!g_vq_cb)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_tex_matmul_vq<<<gr,bl>>>(g_mul,g_add,(const uint8_t*)A,g_tex[tex_idx].tex,g_tex[tex_idx].w,g_vq_cb,(uint8_t*)C,M,K,N);
    return 0;
}
int ari_tex_embed(int tex_idx,const void*ids,void*out,int S,int D){
    if(tex_idx<0||tex_idx>=g_ntex)return-1;
    k_tex_embed<<<S,min(D,BLK)>>>(g_tex[tex_idx].tex,g_tex[tex_idx].w,(const int32_t*)ids,(uint8_t*)out,S,D);
    return 0;
}
int ari_vq_decode_tex(int tex_idx,void*out,int n_blocks){
    if(tex_idx<0||tex_idx>=g_ntex||!g_vq_cb)return-1;
    k_vq_decode_tex<<<(n_blocks+BLK-1)/BLK,BLK>>>(g_tex[tex_idx].tex,g_tex[tex_idx].w,g_vq_cb,(uint8_t*)out,n_blocks);
    return 0;
}
int ari_elem_add(const void*a,const void*b,void*c,int n){
    k_elem_add<<<(n+BLK-1)/BLK,BLK>>>(g_add,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
int ari_elem_mul(const void*a,const void*b,void*c,int n){
    k_elem_mul<<<(n+BLK-1)/BLK,BLK>>>(g_mul,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
int ari_activate(const void*x,void*y,int n){
    k_activate<<<(n+BLK-1)/BLK,BLK>>>(g_cube,(const uint8_t*)x,(uint8_t*)y,n);return 0;
}
int ari_rms_norm(const void*x,void*y,int rows,int cols){
    k_rms_norm<<<rows,BLK>>>(g_mul,g_add,g_inv,(const uint8_t*)x,(uint8_t*)y,rows,cols);return 0;
}
int ari_neg_score(const void*Q,const void*K,void*sc,int B,int H,int S,int T,int Hd){
    int tot=B*H*S*T;
    k_neg_score<<<(tot+BLK-1)/BLK,BLK>>>(g_score,g_add,(const uint8_t*)Q,(const uint8_t*)K,(uint8_t*)sc,B,H,S,T,Hd);return 0;
}
int ari_attn_norm(void*sc,int B,int H,int S,int T){
    int tot=B*H*S;
    k_attn_norm<<<(tot+BLK-1)/BLK,BLK>>>(g_mul,g_add,g_inv,(uint8_t*)sc,B,H,S,T);return 0;
}
int ari_apply_v(const void*sc,const void*V,void*out,int B,int H,int S,int T,int Hd){
    int tot=B*H*S*Hd;
    k_apply_v<<<(tot+BLK-1)/BLK,BLK>>>(g_mul,g_add,(const uint8_t*)sc,(const uint8_t*)V,(uint8_t*)out,B,H,S,T,Hd);return 0;
}
int ari_xpose_bshd(const void*in,void*out,int B,int S,int H,int Hd){
    int tot=B*S*H*Hd;
    k_xpose_bshd_bhsd<<<(tot+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,S,H,Hd);return 0;
}
int ari_xpose_bhsd(const void*in,void*out,int B,int H,int S,int Hd){
    int tot=B*H*S*Hd;
    k_xpose_bhsd_bshd<<<(tot+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,H,S,Hd);return 0;
}
int ari_repeat_kv(const void*in,void*out,int B,int Hkv,int H,int T,int Hd){
    int tot=B*H*T*Hd;
    k_repeat_kv<<<(tot+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,Hkv,H,T,Hd);return 0;
}
int ari_get_tex_w(int idx){return(idx>=0&&idx<g_ntex)?g_tex[idx].w:-1;}
int ari_get_tex_h(int idx){return(idx>=0&&idx<g_ntex)?g_tex[idx].h:-1;}
int ari_tex_matmul_fp16(const void*A,int tex_idx,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_ntex)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_fp16_tex_matmul_t<<<gr,bl>>>((const uint16_t*)A,g_tex[tex_idx].tex,g_tex[tex_idx].w,(uint16_t*)C,M,K,N);
    return 0;
}
int ari_rg_gemv_fused(const void*x,int r_idx,int g_idx,int b_idx,const void*r_lut,const void*g_lut,const void*b_lut,int routing_k,void*y,int N,int K){
    if(r_idx<0||r_idx>=g_ntex||g_idx<0||g_idx>=g_ntex||b_idx<0||b_idx>=g_ntex)return-1;
    k_rg_gemv_fused<<<N,BLK>>>((const uint16_t*)x,g_tex[r_idx].tex,g_tex[r_idx].w,g_tex[g_idx].tex,g_tex[g_idx].w,g_tex[b_idx].tex,g_tex[b_idx].w,(const float*)r_lut,(const float*)g_lut,(const float*)b_lut,routing_k,(uint16_t*)y,N,K);
    return 0;
}
int ari_rg_gemv_fused_offset(const void*x,int r_idx,int g_idx,int b_idx,const void*r_lut,const void*g_lut,const void*b_lut,int routing_k,void*y,int N,int K,int weight_offset){
    if(r_idx<0||r_idx>=g_ntex||g_idx<0||g_idx>=g_ntex||b_idx<0||b_idx>=g_ntex)return-1;
    k_rg_gemv_fused_offset<<<N,BLK>>>((const uint16_t*)x,g_tex[r_idx].tex,g_tex[r_idx].w,g_tex[g_idx].tex,g_tex[g_idx].w,g_tex[b_idx].tex,g_tex[b_idx].w,(const float*)r_lut,(const float*)g_lut,(const float*)b_lut,routing_k,(uint16_t*)y,N,K,weight_offset);
    return 0;
}
int ari_rgba_gemv_fused(const void*x,int f_idx,const void*r_lut,const void*g_lut,const void*b_lut,int routing_k,void*y,int N,int K){
    if(f_idx<0||f_idx>=g_ntex)return-1;
    k_rgba_gemv_fused<<<N,BLK>>>((const uint16_t*)x,g_tex[f_idx].tex,g_tex[f_idx].w,(const float*)r_lut,(const float*)g_lut,(const float*)b_lut,routing_k,(uint16_t*)y,N,K);
    return 0;
}
int ari_gemv_rgba_fp16(const void*x,int W_idx,void*y,int N,int K){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    k_gemv_rgba_fp16<<<N,BLK>>>((const uint16_t*)x,g_tex[W_idx].tex,g_tex[W_idx].w,(uint16_t*)y,N,K);
    return 0;
}
int ari_gemv_rgba_bf16(const void*x,int W_idx,void*y,int N,int K){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    k_gemv_rgba_bf16<<<N,BLK>>>((const uint16_t*)x,g_tex[W_idx].tex,g_tex[W_idx].w,(uint16_t*)y,N,K);
    return 0;
}
int ari_tex_matmul_bf16(const void*A,int tex_idx,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_ntex)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_bf16_tex_matmul_t<<<gr,bl>>>((const uint16_t*)A,g_tex[tex_idx].tex,g_tex[tex_idx].w,(uint16_t*)C,M,K,N);
    return 0;
}
int ari_tex_matmul_fp8(const void*A,int tex_idx,const void*lut_dev,void*C,int M,int K,int N){
    if(tex_idx<0||tex_idx>=g_ntex)return-1;
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_fp8_tex_matmul_t<<<gr,bl>>>((const uint16_t*)A,g_tex[tex_idx].tex,g_tex[tex_idx].w,(const float*)lut_dev,(uint16_t*)C,M,K,N);
    return 0;
}
int ari_gemv_rgba16_fp16(const void*x,int W_idx,void*y,int N,int K){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    if(g_tex[W_idx].fmt!=TEX_FMT_RGBA16)return-2;
    k_gemv_rgba16_fp16<<<N,BLK>>>((const uint16_t*)x,g_tex[W_idx].tex,g_tex[W_idx].w,(uint16_t*)y,N,K);
    return 0;
}
int ari_gemv_ternary_fp16(const void*x,int W_idx,void*y,int N,int K){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    k_gemv_ternary_fp16<<<N,BLK>>>((const uint16_t*)x,g_tex[W_idx].tex,g_tex[W_idx].w,(uint16_t*)y,N,K);
    return 0;
}
int ari_gemv_rtier_fp16(const void*x,int W_idx,const void*scales,void*y,int N,int K,int kpad,int tile_h,int tile_w,int n_tiles_k){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    if(tile_h<=0||tile_w<=0||n_tiles_k<=0||kpad<K)return-2;
    k_gemv_rtier_fp16<<<N,BLK>>>((const uint16_t*)x,g_tex[W_idx].tex,g_tex[W_idx].w,(const uint16_t*)scales,N,K,kpad,tile_h,tile_w,n_tiles_k,(uint16_t*)y);
    return 0;
}
int ari_gemv_rgba16_fp16_tiled(const void*x,int W_idx,void*y,int N,int K,int tile_h,int tile_w){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    if(g_tex[W_idx].fmt!=TEX_FMT_RGBA16)return-2;
    if(tile_h<=0||tile_w<=0||(tile_w&3))return-3;
    k_gemv_rgba16_fp16_tiled<<<N,BLK>>>((const uint16_t*)x,g_tex[W_idx].tex,tile_h,tile_w,(uint16_t*)y,N,K);
    return 0;
}
int ari_fetch_rgba16_indices(int tex_idx,const void*indices,void*out,int n){
    if(tex_idx<0||tex_idx>=g_ntex)return-1;
    if(g_tex[tex_idx].fmt!=TEX_FMT_RGBA16)return-2;
    if(n<0)return-3;
    k_fetch_rgba16_indices<<<(n+BLK-1)/BLK,BLK>>>(g_tex[tex_idx].tex,g_tex[tex_idx].w,(const int32_t*)indices,(uint16_t*)out,n);
    return 0;
}
static int disp_grow(int n){
    if(n<=g_disp_cap)return 0;
    int nc=n<64?64:(n*2);
    if(g_disp_gu_h){hipFree(g_disp_gu_h);hipFree(g_disp_dn_h);hipFree(g_disp_gu_tw);hipFree(g_disp_dn_tw);hipFree(g_disp_gu_fmt);hipFree(g_disp_dn_fmt);}
    g_disp_gu_h=g_disp_dn_h=nullptr;g_disp_gu_tw=g_disp_dn_tw=g_disp_gu_fmt=g_disp_dn_fmt=nullptr;
    if(hipMalloc(&g_disp_gu_h,(size_t)nc*sizeof(uint64_t))!=hipSuccess)return-1;
    if(hipMalloc(&g_disp_dn_h,(size_t)nc*sizeof(uint64_t))!=hipSuccess)return-1;
    if(hipMalloc(&g_disp_gu_tw,(size_t)nc*sizeof(int32_t))!=hipSuccess)return-1;
    if(hipMalloc(&g_disp_dn_tw,(size_t)nc*sizeof(int32_t))!=hipSuccess)return-1;
    if(hipMalloc(&g_disp_gu_fmt,(size_t)nc*sizeof(int32_t))!=hipSuccess)return-1;
    if(hipMalloc(&g_disp_dn_fmt,(size_t)nc*sizeof(int32_t))!=hipSuccess)return-1;
    g_disp_cap=nc;return 0;
}
int ari_dispatch_fused_rgba16(
    const void* x_in, void* y_out,
    int num_tokens, int hidden,
    int num_routes, const int32_t* token_ids, const float* weights,
    const int32_t* gu_tex_idxs_host, const int32_t* dn_tex_idxs_host,
    const int32_t* gu_offsets, const int32_t* dn_offsets,
    int N_gu, int K_gu, int N_dn, int K_dn
){
    if(num_routes<=0||num_tokens<=0||hidden<=0)return 0;
    if(N_dn!=hidden||K_gu!=hidden||N_gu!=2*K_dn)return-1;
    size_t n=(size_t)num_routes;
    std::vector<uint64_t> gu_h(n),dn_h(n);
    std::vector<int32_t> gu_tw(n),dn_tw(n),gu_fmt(n),dn_fmt(n);
    for(size_t i=0;i<n;i++){
        int gi=gu_tex_idxs_host[i],di=dn_tex_idxs_host[i];
        if(gi<0||gi>=g_ntex||di<0||di>=g_ntex)return-2;
        if(g_tex[gi].arr==nullptr||g_tex[di].arr==nullptr)return-3;
        gu_h[i]=(uint64_t)g_tex[gi].tex;dn_h[i]=(uint64_t)g_tex[di].tex;
        gu_tw[i]=g_tex[gi].w;dn_tw[i]=g_tex[di].w;
        gu_fmt[i]=g_tex[gi].fmt;dn_fmt[i]=g_tex[di].fmt;
    }
    if(disp_grow(num_routes)!=0)return-4;
    hipMemcpy(g_disp_gu_h,gu_h.data(),n*sizeof(uint64_t),hipMemcpyHostToDevice);
    hipMemcpy(g_disp_dn_h,dn_h.data(),n*sizeof(uint64_t),hipMemcpyHostToDevice);
    hipMemcpy(g_disp_gu_tw,gu_tw.data(),n*sizeof(int32_t),hipMemcpyHostToDevice);
    hipMemcpy(g_disp_dn_tw,dn_tw.data(),n*sizeof(int32_t),hipMemcpyHostToDevice);
    hipMemcpy(g_disp_gu_fmt,gu_fmt.data(),n*sizeof(int32_t),hipMemcpyHostToDevice);
    hipMemcpy(g_disp_dn_fmt,dn_fmt.data(),n*sizeof(int32_t),hipMemcpyHostToDevice);
    size_t smem=((size_t)K_dn+(size_t)K_gu)*sizeof(float);
    if(smem>96*1024)return-5;
    k_dispatch_fused_routes<<<num_routes,BLK,smem>>>(
        (const uint16_t*)x_in,
        token_ids,weights,
        g_disp_gu_h,g_disp_gu_tw,g_disp_gu_fmt,gu_offsets,
        g_disp_dn_h,g_disp_dn_tw,g_disp_dn_fmt,dn_offsets,
        (float*)y_out,num_tokens,hidden,N_gu,K_gu,N_dn,K_dn);
    return 0;
}
int ari_gemv_rgba16_fp16_batched(const void*X,int W_idx,void*Y,int M,int N,int K){
    if(W_idx<0||W_idx>=g_ntex)return-1;
    if(g_tex[W_idx].fmt!=TEX_FMT_RGBA16)return-2;
    if(M<=0||N<=0||K<=0)return-3;
    dim3 gr((unsigned)N,(unsigned)M,1);
    k_gemv_rgba16_fp16_batched<<<gr,BLK>>>((const uint16_t*)X,g_tex[W_idx].tex,g_tex[W_idx].w,(uint16_t*)Y,M,N,K);
    return 0;
}
}
