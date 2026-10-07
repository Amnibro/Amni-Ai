import os,sys,json,torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amni.serve import blender_atex as B
0 if os.environ.get("NOFIX") else B.noexp();torch.manual_seed(0);dev="cuda";R=[];sizes=[int(x) for x in torch.randint(1<<16,1<<25,(160,)).tolist()]
for rep in range(int(sys.argv[1]) if len(sys.argv)>1 else 8):
    H=[torch.randint(-120,120,(n,),dtype=torch.int8) for n in sizes[:40+rep*7%60]];G=[]
    for h in H:g=torch.empty(h.shape,dtype=torch.int8,device=dev);g.copy_(h);G.append(g)
    t=[torch.randn(1<<22,device=dev) for _ in range(8)];del t
    torch.cuda.synchronize();bad=sum(int((g.cpu()!=h).any()) for g,h in zip(G,H));R.append(bad);print("rep",rep,"tensors",len(G),"GB",round(sum(g.numel() for g in G)/1e9,2),"corrupt",bad,flush=True)
    del G,H;torch.cuda.empty_cache()
print(json.dumps({"alloc_conf":os.environ.get("PYTORCH_HIP_ALLOC_CONF",""),"fix":not os.environ.get("NOFIX"),"corrupt_per_rep":R}))
