#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
__global__ void k_wmma_gemm_16x16x16(const _Float16* A, const _Float16* B, _Float16* C){
    int lane=threadIdx.x;
    v16h a, bv, c;
    for(int i=0;i<16;i++)a[i]=A[(lane%16)*16+i];
    for(int i=0;i<16;i++)bv[i]=B[i*16+(lane%16)];
    for(int i=0;i<16;i++)c[i]=(_Float16)0;
    c=__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, bv, c, false);
    for(int i=0;i<8;i++)C[(2*i+(lane/16))*16+lane%16]=c[i*2];
}
#define EXPORT __declspec(dllexport)
extern "C"{
EXPORT int wmma_gemm_init(int dev){return hipSetDevice(dev)==hipSuccess?0:-1;}
EXPORT void* wmma_gemm_alloc(size_t n){void* p=nullptr; hipMalloc(&p,n); return p;}
EXPORT void wmma_gemm_free(void* p){if(p)hipFree(p);}
EXPORT int wmma_gemm_h2d(void* d, const void* s, size_t n){return hipMemcpy(d,s,n,hipMemcpyHostToDevice)==hipSuccess?0:-1;}
EXPORT int wmma_gemm_d2h(void* d, const void* s, size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToHost)==hipSuccess?0:-1;}
EXPORT int wmma_gemm_sync(){return hipDeviceSynchronize()==hipSuccess?0:-1;}
EXPORT int wmma_gemm_run(const void* A, const void* B, void* C){
    hipLaunchKernelGGL(k_wmma_gemm_16x16x16, dim3(1), dim3(32), 0, 0,
        (const _Float16*)A, (const _Float16*)B, (_Float16*)C);
    return hipGetLastError()==hipSuccess?0:-1;
}
}
