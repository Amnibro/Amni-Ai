#include <hip/hip_runtime.h>
#include <cstdint>
#define P 17
#define P2 (P*P)
#define TILE 16
#define BLK 256
static uint8_t *g_mul=nullptr,*g_add=nullptr,*g_score=nullptr;
static uint8_t *g_inv=nullptr,*g_cube=nullptr;
static bool g_ok=false;
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
__global__ void k_matmul_t(
    const uint8_t*__restrict__ ml,const uint8_t*__restrict__ al,
    const uint8_t*__restrict__ A,const uint8_t*__restrict__ W,
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
        Ws[threadIdx.y][threadIdx.x]=(col<N&&wk<K)?W[col*K+wk]:0;
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc=sa[acc*P+sm[As[threadIdx.y][k]*P+Ws[k][threadIdx.x]]];
        }
        __syncthreads();
    }
    if(row<M&&col<N)C[row*N+col]=acc;
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
__global__ void k_embed(const uint8_t*__restrict__ emb,const int32_t*__restrict__ ids,uint8_t*__restrict__ out,int S,int D){
    int s=blockIdx.x;if(s>=S)return;
    for(int d=threadIdx.x;d<D;d+=blockDim.x)out[s*D+d]=emb[ids[s]*D+d];
}
__global__ void k_unpack2b_lut(const uint8_t*__restrict__ packed,const uint8_t*__restrict__ lut,uint8_t*__restrict__ out,int n){
    __shared__ uint8_t sl[4];
    if(threadIdx.x<4)sl[threadIdx.x]=lut[threadIdx.x];
    __syncthreads();
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n)return;
    int bi=tid>>2,si=tid&3;
    out[tid]=sl[(packed[bi]>>(si*2))&3];
}
#define GF17X
extern "C"{
GF17X int gf17_init(int dev){
    if(g_ok)return 0;
    if(hipSetDevice(dev)!=hipSuccess)return-1;
    uint8_t hm[P2],ha[P2],hs[P2],hi[P],hc[P];
    compute_tables(hm,ha,hs,hi,hc);
    hipMalloc(&g_mul,P2);hipMemcpy(g_mul,hm,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_add,P2);hipMemcpy(g_add,ha,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_score,P2);hipMemcpy(g_score,hs,P2,hipMemcpyHostToDevice);
    hipMalloc(&g_inv,P);hipMemcpy(g_inv,hi,P,hipMemcpyHostToDevice);
    hipMalloc(&g_cube,P);hipMemcpy(g_cube,hc,P,hipMemcpyHostToDevice);
    g_ok=true;return 0;
}
GF17X void gf17_shutdown(){
    if(!g_ok)return;
    hipFree(g_mul);hipFree(g_add);hipFree(g_score);hipFree(g_inv);hipFree(g_cube);
    g_mul=g_add=g_score=g_inv=g_cube=nullptr;g_ok=false;
}
GF17X void*gf17_alloc(size_t n){void*p=nullptr;hipMalloc(&p,n);return p;}
GF17X void gf17_free(void*p){if(p)hipFree(p);}
GF17X int gf17_h2d(void*d,const void*s,size_t n){return hipMemcpy(d,s,n,hipMemcpyHostToDevice)==hipSuccess?0:-1;}
GF17X int gf17_d2h(void*d,const void*s,size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToHost)==hipSuccess?0:-1;}
GF17X int gf17_d2d(void*d,const void*s,size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToDevice)==hipSuccess?0:-1;}
GF17X int gf17_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
GF17X int gf17_matmul_t(const void*A,const void*W,void*C,int M,int K,int N){
    dim3 bl(TILE,TILE),gr((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    k_matmul_t<<<gr,bl>>>(g_mul,g_add,(const uint8_t*)A,(const uint8_t*)W,(uint8_t*)C,M,K,N);return 0;
}
GF17X int gf17_elem_add(const void*a,const void*b,void*c,int n){
    k_elem_add<<<(n+BLK-1)/BLK,BLK>>>(g_add,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
GF17X int gf17_elem_mul(const void*a,const void*b,void*c,int n){
    k_elem_mul<<<(n+BLK-1)/BLK,BLK>>>(g_mul,(const uint8_t*)a,(const uint8_t*)b,(uint8_t*)c,n);return 0;
}
GF17X int gf17_activate(const void*x,void*y,int n){
    k_activate<<<(n+BLK-1)/BLK,BLK>>>(g_cube,(const uint8_t*)x,(uint8_t*)y,n);return 0;
}
GF17X int gf17_rms_norm(const void*x,void*y,int rows,int cols){
    k_rms_norm<<<rows,BLK>>>(g_mul,g_add,g_inv,(const uint8_t*)x,(uint8_t*)y,rows,cols);return 0;
}
GF17X int gf17_neg_score(const void*Q,const void*K,void*sc,int B,int H,int S,int T,int Hd){
    int tot=B*H*S*T;
    k_neg_score<<<(tot+BLK-1)/BLK,BLK>>>(g_score,g_add,(const uint8_t*)Q,(const uint8_t*)K,(uint8_t*)sc,B,H,S,T,Hd);return 0;
}
GF17X int gf17_attn_norm(void*sc,int B,int H,int S,int T){
    int tot=B*H*S;
    k_attn_norm<<<(tot+BLK-1)/BLK,BLK>>>(g_mul,g_add,g_inv,(uint8_t*)sc,B,H,S,T);return 0;
}
GF17X int gf17_apply_v(const void*sc,const void*V,void*out,int B,int H,int S,int T,int Hd){
    int tot=B*H*S*Hd;
    k_apply_v<<<(tot+BLK-1)/BLK,BLK>>>(g_mul,g_add,(const uint8_t*)sc,(const uint8_t*)V,(uint8_t*)out,B,H,S,T,Hd);return 0;
}
GF17X int gf17_xpose_bshd(const void*in,void*out,int B,int S,int H,int Hd){
    int tot=B*S*H*Hd;
    k_xpose_bshd_bhsd<<<(tot+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,S,H,Hd);return 0;
}
GF17X int gf17_xpose_bhsd(const void*in,void*out,int B,int H,int S,int Hd){
    int tot=B*H*S*Hd;
    k_xpose_bhsd_bshd<<<(tot+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,H,S,Hd);return 0;
}
GF17X int gf17_repeat_kv(const void*in,void*out,int B,int Hkv,int H,int T,int Hd){
    int tot=B*H*T*Hd;
    k_repeat_kv<<<(tot+BLK-1)/BLK,BLK>>>((const uint8_t*)in,(uint8_t*)out,B,Hkv,H,T,Hd);return 0;
}
GF17X int gf17_embed(const void*emb,const void*ids,void*out,int S,int D){
    int th=D<BLK?D:BLK;
    k_embed<<<S,th>>>((const uint8_t*)emb,(const int32_t*)ids,(uint8_t*)out,S,D);return 0;
}
GF17X int gf17_unpack2b(const void*packed,const void*lut,void*out,int n){
    k_unpack2b_lut<<<(n+BLK-1)/BLK,BLK>>>((const uint8_t*)packed,(const uint8_t*)lut,(uint8_t*)out,n);return 0;
}
#define F32BLK 256
__global__ void k_dq_gemv_f32(const uint8_t*__restrict__ W,const float*__restrict__ x,float*__restrict__ y,int K,int N,float scale){
    int n=blockIdx.x;if(n>=N)return;
    int tid=threadIdx.x;
    __shared__ float ps[F32BLK];
    float acc=0.0f;
    const uint8_t*row=W+(size_t)n*K;
    for(int k=tid;k<K;k+=F32BLK)acc+=((float)row[k]*0.125f-1.0f)*scale*x[k];
    ps[tid]=acc;__syncthreads();
    for(int s=F32BLK/2;s>0;s>>=1){if(tid<s)ps[tid]+=ps[tid+s];__syncthreads();}
    if(tid==0)y[n]=ps[0];
}
__global__ void k_rms_norm_f32(const float*__restrict__ x,float*__restrict__ y,int rows,int cols,float eps){
    int row=blockIdx.x;if(row>=rows)return;
    int tid=threadIdx.x;
    __shared__ float ps[F32BLK];
    const float*xr=x+row*cols;float*yr=y+row*cols;
    float acc=0.0f;
    for(int j=tid;j<cols;j+=F32BLK)acc+=xr[j]*xr[j];
    ps[tid]=acc;__syncthreads();
    for(int s=F32BLK/2;s>0;s>>=1){if(tid<s)ps[tid]+=ps[tid+s];__syncthreads();}
    float inv=rsqrtf(ps[0]/(float)cols+eps);
    for(int j=tid;j<cols;j+=F32BLK)yr[j]=xr[j]*inv;
}
__global__ void k_elem_add_f32(const float*__restrict__ a,const float*__restrict__ b,float*__restrict__ c,int n){
    int i=blockIdx.x*F32BLK+threadIdx.x;if(i<n)c[i]=a[i]+b[i];
}
__global__ void k_silu_inp_f32(const float*__restrict__ g,const float*__restrict__ u,float*__restrict__ o,int n){
    int i=blockIdx.x*F32BLK+threadIdx.x;if(i>=n)return;
    float gv=g[i];o[i]=gv/(1.0f+expf(-gv))*u[i];
}
GF17X void*gf17_alloc_f32(size_t n){void*p=nullptr;hipMalloc(&p,n*sizeof(float));return p;}
GF17X int gf17_h2d_f32(void*d,const void*s,size_t n){return hipMemcpy(d,s,n*sizeof(float),hipMemcpyHostToDevice)==hipSuccess?0:-1;}
GF17X int gf17_d2h_f32(void*d,const void*s,size_t n){return hipMemcpy(d,s,n*sizeof(float),hipMemcpyDeviceToHost)==hipSuccess?0:-1;}
GF17X int gf17_dq_gemv_f32(const void*W,const void*x,void*y,int K,int N,float scale){
    k_dq_gemv_f32<<<N,F32BLK>>>((const uint8_t*)W,(const float*)x,(float*)y,K,N,scale);return 0;}
__global__ void k_dq_gemv_b17_f32(const uint8_t*__restrict__ W,const float*__restrict__ x,float*__restrict__ y,int K,int N){
    int n=blockIdx.x;if(n>=N)return;
    int tid=threadIdx.x;
    __shared__ float ps[F32BLK];
    float acc=0.0f;
    const uint8_t*row=W+(size_t)n*K*4;
    for(int k=tid;k<K;k+=F32BLK){
        uint32_t u16=(uint32_t)row[4*k]+(uint32_t)row[4*k+1]*17u+(uint32_t)row[4*k+2]*289u+(uint32_t)row[4*k+3]*4913u;
        float val=__uint_as_float(u16<<16);
        acc+=val*x[k];
    }
    ps[tid]=acc;__syncthreads();
    for(int s=F32BLK/2;s>0;s>>=1){if(tid<s)ps[tid]+=ps[tid+s];__syncthreads();}
    if(tid==0)y[n]=ps[0];
}
GF17X int gf17_dq_gemv_b17_f32(const void*W,const void*x,void*y,int K,int N){
    k_dq_gemv_b17_f32<<<N,F32BLK>>>((const uint8_t*)W,(const float*)x,(float*)y,K,N);return 0;}
GF17X int gf17_rms_norm_f32(const void*x,void*y,int rows,int cols,float eps){
    k_rms_norm_f32<<<rows,F32BLK>>>((const float*)x,(float*)y,rows,cols,eps);return 0;}
GF17X int gf17_elem_add_f32(const void*a,const void*b,void*c,int n){
    k_elem_add_f32<<<(n+F32BLK-1)/F32BLK,F32BLK>>>((const float*)a,(const float*)b,(float*)c,n);return 0;}
GF17X int gf17_silu_inp_f32(const void*g,const void*u,void*o,int n){
    k_silu_inp_f32<<<(n+F32BLK-1)/F32BLK,F32BLK>>>((const float*)g,(const float*)u,(float*)o,n);return 0;}
__global__ void k_attn_score_f32(const float*Q,const float*K,float*sc,int H,int Hkv,int Hd,int T,float inv_sqrt){
    int idx=blockIdx.x*F32BLK+threadIdx.x;if(idx>=H*T)return;
    int h=idx/T,t=idx%T,hkv=h*Hkv/H;
    float s=0.0f;const float*qr=Q+h*Hd;const float*kr=K+(t*Hkv+hkv)*Hd;
    for(int d=0;d<Hd;d++)s+=qr[d]*kr[d];sc[idx]=s*inv_sqrt;}
__global__ void k_softmax_rows_f32(float*sc,int H,int T){
    int h=blockIdx.x;if(h>=H)return;float*row=sc+h*T;int tid=threadIdx.x;
    __shared__ float ps[F32BLK];
    float mx=-1e38f;for(int t=tid;t<T;t+=F32BLK)mx=fmaxf(mx,row[t]);
    ps[tid]=mx;__syncthreads();
    for(int s=F32BLK/2;s>0;s>>=1){if(tid<s)ps[tid]=fmaxf(ps[tid],ps[tid+s]);__syncthreads();}
    mx=ps[0];float sum=0.0f;
    for(int t=tid;t<T;t+=F32BLK){row[t]=expf(row[t]-mx);sum+=row[t];}
    ps[tid]=sum;__syncthreads();
    for(int s=F32BLK/2;s>0;s>>=1){if(tid<s)ps[tid]+=ps[tid+s];__syncthreads();}
    float inv=1.0f/ps[0];for(int t=tid;t<T;t+=F32BLK)row[t]*=inv;}
__global__ void k_attn_out_f32(const float*sc,const float*V,float*out,int H,int Hkv,int Hd,int T){
    int idx=blockIdx.x*F32BLK+threadIdx.x;if(idx>=H*Hd)return;
    int h=idx/Hd,d=idx%Hd,hkv=h*Hkv/H;
    float s=0.0f;const float*scr=sc+h*T;
    for(int t=0;t<T;t++)s+=scr[t]*V[(t*Hkv+hkv)*Hd+d];out[idx]=s;}
GF17X int gf17_mqa_attn_f32(const void*Q,void*KC,void*VC,const void*nk,const void*nv,void*sb,void*out,int H,int Hkv,int Hd,int T,float inv_sqrt){
    int kv_n=Hkv*Hd;
    hipMemcpy((float*)KC+(T-1)*kv_n,nk,kv_n*sizeof(float),hipMemcpyDeviceToDevice);
    hipMemcpy((float*)VC+(T-1)*kv_n,nv,kv_n*sizeof(float),hipMemcpyDeviceToDevice);
    k_attn_score_f32<<<(H*T+F32BLK-1)/F32BLK,F32BLK>>>((const float*)Q,(const float*)KC,(float*)sb,H,Hkv,Hd,T,inv_sqrt);
    k_softmax_rows_f32<<<H,F32BLK>>>((float*)sb,H,T);
    k_attn_out_f32<<<(H*Hd+F32BLK-1)/F32BLK,F32BLK>>>((const float*)sb,(const float*)VC,(float*)out,H,Hkv,Hd,T);
    return 0;}
}
