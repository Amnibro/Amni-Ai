#include <hip/hip_runtime.h>
#include <cstdint>
#define P 17
#define P2 (P*P)
#define P3 (P*P*P)
#define P4 (P*P*P*P)
#define BLK 256
#define PTEX_MAGIC 0x50545558
__device__ __forceinline__ uint8_t ptex_raw4(const uint8_t*px,int sub_idx){return px[sub_idx];}
__device__ __forceinline__ uint8_t ptex_base17_4(const uint8_t*px,int sub_idx){return px[sub_idx];}
__device__ __forceinline__ uint16_t ptex_fp16_from_base17(const uint8_t*px){
    return (uint16_t)(px[0]+px[1]*P+px[2]*P2+px[3]*P3);
}
__device__ __forceinline__ void ptex_decode_dense16(
    const uint8_t*map0,const uint8_t*map1,const uint8_t*map2,const uint8_t*map3,
    uint8_t*out){
    #pragma unroll
    for(int c=0;c<4;c++){out[c]=map0[c];out[4+c]=map1[c];out[8+c]=map2[c];out[12+c]=map3[c];}
}
__device__ __forceinline__ uint8_t ptex_lut_fetch(
    hipTextureObject_t lut_tex,int lut_w,uint8_t a,uint8_t b){
    int flat=a*P+b;
    int py=flat/(lut_w*4);
    int px_r=flat%(lut_w*4);
    int px_c=px_r/4,ch=px_r%4;
    uchar4 texel=tex2D<uchar4>(lut_tex,(float)px_c+0.5f,(float)py+0.5f);
    uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
    return vals[ch];
}
__global__ void k_ptex_decode_raw4(
    hipTextureObject_t primary_tex,int tex_w,
    uint8_t*__restrict__ out,int n_weights){
    int wi=blockIdx.x*blockDim.x+threadIdx.x;
    if(wi>=n_weights)return;
    int px_idx=wi/4,sub=wi%4;
    int py=px_idx/(tex_w);
    int px=px_idx%(tex_w);
    uchar4 texel=tex2D<uchar4>(primary_tex,(float)px+0.5f,(float)py+0.5f);
    uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
    out[wi]=vals[sub];
}
__global__ void k_ptex_decode_fp16(
    hipTextureObject_t primary_tex,int tex_w,
    uint16_t*__restrict__ out,int n_weights){
    int wi=blockIdx.x*blockDim.x+threadIdx.x;
    if(wi>=n_weights)return;
    int py=wi/tex_w,px=wi%tex_w;
    uchar4 texel=tex2D<uchar4>(primary_tex,(float)px+0.5f,(float)py+0.5f);
    out[wi]=(uint16_t)(texel.x+texel.y*P+texel.z*P2+texel.w*P3);
}
__global__ void k_ptex_decode_dense16(
    hipTextureObject_t tex0,hipTextureObject_t tex1,
    hipTextureObject_t tex2,hipTextureObject_t tex3,
    int tex_w,uint8_t*__restrict__ out,int n_groups){
    int gi=blockIdx.x*blockDim.x+threadIdx.x;
    if(gi>=n_groups)return;
    int py=gi/tex_w,px=gi%tex_w;
    uchar4 t0=tex2D<uchar4>(tex0,(float)px+0.5f,(float)py+0.5f);
    uchar4 t1=tex2D<uchar4>(tex1,(float)px+0.5f,(float)py+0.5f);
    uchar4 t2=tex2D<uchar4>(tex2,(float)px+0.5f,(float)py+0.5f);
    uchar4 t3=tex2D<uchar4>(tex3,(float)px+0.5f,(float)py+0.5f);
    int base=gi*16;
    out[base]=t0.x;out[base+1]=t0.y;out[base+2]=t0.z;out[base+3]=t0.w;
    out[base+4]=t1.x;out[base+5]=t1.y;out[base+6]=t1.z;out[base+7]=t1.w;
    out[base+8]=t2.x;out[base+9]=t2.y;out[base+10]=t2.z;out[base+11]=t2.w;
    out[base+12]=t3.x;out[base+13]=t3.y;out[base+14]=t3.z;out[base+15]=t3.w;
}
__global__ void k_ptex_lut_matmul(
    const uint8_t*__restrict__ mul_lut,const uint8_t*__restrict__ add_lut,
    hipTextureObject_t W_tex,int tex_w,
    const uint8_t*__restrict__ A,
    uint8_t*__restrict__ C,int M,int K,int N,int mode){
    __shared__ uint8_t sm[P2],sa[P2],As[16][16],Ws[16][16];
    int tid=threadIdx.y*16+threadIdx.x;
    for(int i=tid;i<P2;i+=256){sm[i]=mul_lut[i];sa[i]=add_lut[i];}
    __syncthreads();
    int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*16+threadIdx.x;
    uint8_t acc=0;
    for(int t=0;t<(K+15)/16;t++){
        int ak=t*16+threadIdx.x,wk=t*16+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0;
        if(col<N&&wk<K){
            int w_flat=col*K+wk;
            int px_idx=w_flat/4,sub=w_flat%4;
            int py=px_idx/tex_w,px=px_idx%tex_w;
            uchar4 texel=tex2D<uchar4>(W_tex,(float)px+0.5f,(float)py+0.5f);
            uint8_t vals[4]={texel.x,texel.y,texel.z,texel.w};
            Ws[threadIdx.y][threadIdx.x]=vals[sub];
        }else{Ws[threadIdx.y][threadIdx.x]=0;}
        __syncthreads();
        #pragma unroll
        for(int k=0;k<16;k++){
            acc=sa[acc*P+sm[As[threadIdx.y][k]*P+Ws[k][threadIdx.x]]];
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void k_ptex_nonce_lut_decode(
    hipTextureObject_t primary_tex,int tex_w,
    const uint8_t*__restrict__ lut,int lut_size,
    uint8_t*__restrict__ out,int n_weights){
    int wi=blockIdx.x*blockDim.x+threadIdx.x;
    if(wi>=n_weights)return;
    int py=wi/tex_w,px=wi%tex_w;
    uchar4 texel=tex2D<uchar4>(primary_tex,(float)px+0.5f,(float)py+0.5f);
    uint32_t idx=(uint32_t)texel.x+(uint32_t)texel.y*P+(uint32_t)texel.z*P2+(uint32_t)texel.w*P3;
    out[wi]=lut[idx%lut_size];
}
__global__ void k_ptex_micro_lut_decode(
    hipTextureObject_t primary_tex,int tex_w,
    const uint8_t*__restrict__ lut,
    uint8_t*__restrict__ out,int n_weights){
    int wi=blockIdx.x*blockDim.x+threadIdx.x;
    if(wi>=n_weights)return;
    int py=wi/tex_w,px=wi%tex_w;
    uchar4 texel=tex2D<uchar4>(primary_tex,(float)px+0.5f,(float)py+0.5f);
    uint32_t idx=((uint32_t)texel.x+(uint32_t)texel.y*P+(uint32_t)texel.z*P2)%256;
    out[wi]=lut[idx];
}
extern "C"{
static uint8_t*g_mul=nullptr,*g_add=nullptr;
static bool g_ptex_ok=false;
int ptex_init(int dev){
    if(g_ptex_ok)return 0;
    if(hipSetDevice(dev)!=hipSuccess)return-1;
    uint8_t hm[P2],ha[P2];
    for(int i=0;i<P;i++)for(int j=0;j<P;j++){hm[i*P+j]=(i*j)%P;ha[i*P+j]=(i+j)%P;}
    hipMalloc(&g_mul,P2);hipMemcpy(g_mul,hm,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_add,P2);hipMemcpy(g_add,ha,P2,hipMemcpyHostToDevice);
    g_ptex_ok=true;return 0;
}
void ptex_shutdown(){
    if(!g_ptex_ok)return;
    hipFree(g_mul);hipFree(g_add);g_mul=g_add=nullptr;g_ptex_ok=false;
}
int ptex_decode_raw4(hipTextureObject_t tex,int tex_w,void*out,int n){
    k_ptex_decode_raw4<<<(n+BLK-1)/BLK,BLK>>>(tex,tex_w,(uint8_t*)out,n);return 0;
}
int ptex_decode_fp16(hipTextureObject_t tex,int tex_w,void*out,int n){
    k_ptex_decode_fp16<<<(n+BLK-1)/BLK,BLK>>>(tex,tex_w,(uint16_t*)out,n);return 0;
}
int ptex_decode_dense16(hipTextureObject_t t0,hipTextureObject_t t1,hipTextureObject_t t2,hipTextureObject_t t3,int tex_w,void*out,int n_groups){
    k_ptex_decode_dense16<<<(n_groups+BLK-1)/BLK,BLK>>>(t0,t1,t2,t3,tex_w,(uint8_t*)out,n_groups);return 0;
}
int ptex_lut_matmul(hipTextureObject_t W_tex,int tex_w,const void*A,void*C,int M,int K,int N,int mode){
    dim3 bl(16,16),gr((N+15)/16,(M+15)/16);
    k_ptex_lut_matmul<<<gr,bl>>>(g_mul,g_add,W_tex,tex_w,(const uint8_t*)A,(uint8_t*)C,M,K,N,mode);return 0;
}
int ptex_nonce_lut_decode(hipTextureObject_t tex,int tex_w,const void*lut,int lut_size,void*out,int n){
    k_ptex_nonce_lut_decode<<<(n+BLK-1)/BLK,BLK>>>(tex,tex_w,(const uint8_t*)lut,lut_size,(uint8_t*)out,n);return 0;
}
int ptex_micro_lut_decode(hipTextureObject_t tex,int tex_w,const void*lut,void*out,int n){
    k_ptex_micro_lut_decode<<<(n+BLK-1)/BLK,BLK>>>(tex,tex_w,(const uint8_t*)lut,(uint8_t*)out,n);return 0;
}
}
