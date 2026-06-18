import torch,torch.nn as nn
class MonarchLinear(nn.Module):
    def __init__(self,in_features,out_features,m,p,bias=False,device=None,dtype=torch.float32):
        super().__init__()
        assert in_features%m==0 and out_features%p==0
        self.in_features=in_features;self.out_features=out_features;self.m=m;self.p=p;self.k1=in_features//m;self.o=out_features//p
        self.R=nn.Parameter(torch.randn(m,self.k1,p,device=device,dtype=dtype)/(self.k1**0.5))
        self.L=nn.Parameter(torch.randn(p,m,self.o,device=device,dtype=dtype)/(m**0.5))
        self.bias=nn.Parameter(torch.zeros(out_features,device=device,dtype=dtype)) if bias else None
    def forward(self,x):
        B=x.shape[:-1];t=torch.einsum('...mk,mkp->...pm',x.reshape(*B,self.m,self.k1),self.R)
        y=torch.einsum('...pm,pmo->...po',t,self.L).reshape(*B,self.out_features)
        return y+self.bias if self.bias is not None else y
    def materialize(self):return torch.einsum('pmo,mkp->pomk',self.L,self.R).reshape(self.out_features,self.in_features)
    def param_count(self):return self.R.numel()+self.L.numel()
    def reduction(self):return self.in_features*self.out_features/self.param_count()
