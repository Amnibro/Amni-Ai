import torch,triton,triton.language as tl
print('torch:',torch.__version__,'triton:',triton.__version__)
@triton.jit
def add_kernel(x_ptr,y_ptr,out_ptr,n,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    off=pid*BLOCK+tl.arange(0,BLOCK)
    mask=off<n
    x=tl.load(x_ptr+off,mask=mask)
    y=tl.load(y_ptr+off,mask=mask)
    tl.store(out_ptr+off,x+y,mask=mask)
def main():
    n=4096
    x=torch.randn(n,device='cuda',dtype=torch.float32)
    y=torch.randn(n,device='cuda',dtype=torch.float32)
    out=torch.empty_like(x)
    grid=lambda meta:(triton.cdiv(n,meta['BLOCK']),)
    add_kernel[grid](x,y,out,n,BLOCK=256)
    ref=x+y
    diff=(out-ref).abs().max().item()
    print(f'triton add OK max_diff={diff:.6e}')
if __name__=='__main__':main()
