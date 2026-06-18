"""Gated density pages — runtime-addressed weight learning (the proven densify_gated recipe, productionized).
Each learning = a low-rank residual page on late-MLP layers + a Reffelt-nonce address key (mean centered residual-stream hidden of its prompts).
gate=sigmoid((cos(x-mu,key)-TAU)*SHARP) -> the page speaks ONLY when a query routes near its address: plasticity on-domain, zero collateral off-domain, additive across independent pages, exactly removable. Train gate-open (strong memorize), serve sharp-gated."""
import torch,torch.nn as nn,torch.nn.functional as F
GENERIC=["The capital of France is Paris.","Water is made of hydrogen and oxygen.","Two plus two equals four.","The sun rises in the east.","A dog is a kind of animal.","The novel explores themes of love and loss.","Photosynthesis converts sunlight into energy.","Newton described the laws of motion.","The weather today is sunny and warm.","DNA carries genetic information.","Shakespeare wrote many famous plays.","The ocean covers most of the planet."]
class _GatedMLP(nn.Module):
    def __init__(s,mlp,hs,tau,sharp):
        super().__init__();s.base=mlp;s.hs=hs;s.tau=tau;s.sharp=sharp;s.pg=nn.ParameterList();s.keys=[];s.mus=[];s.on=[];s.force=[];s.last=None
        for p in s.base.parameters():p.requires_grad_(False)
    def add(s,key,muv,r):
        A=nn.Parameter(torch.zeros(r,s.hs,device=key.device,dtype=torch.float32));B=nn.Parameter(torch.randn(s.hs,r,device=key.device,dtype=torch.float32)*1e-3)
        s.pg.append(A);s.pg.append(B);s.keys.append(key);s.mus.append(muv);s.on.append(True);s.force.append(False);return len(s.keys)-1
    def forward(s,x):
        s.last=x.detach();y=s.base(x);xf=x.float()
        for j in range(len(s.keys)):
            if s.on[j]:
                c=(xf@s.pg[2*j].t())@s.pg[2*j+1].t()
                y=y+c.to(y.dtype) if s.force[j] else y+(torch.sigmoid((F.cosine_similarity(xf-s.mus[j],s.keys[j].expand_as(xf),dim=-1)-s.tau)*s.sharp).unsqueeze(-1)*c).to(y.dtype)
        return y
class GatedPageBank:
    def __init__(s,model,tok,layers=None,r=16,tau=0.28,sharp=22):
        s.model=model;s.tok=tok;s.r=r;s.layers=layers or list(range(16,28,2));s.mu=None;s.domains={}
        hs=model.config.hidden_size if hasattr(model.config,'hidden_size') else model.config.get_text_config().hidden_size
        s.mods={}
        for li in s.layers:
            w=_GatedMLP(model.model.layers[li].mlp,hs,tau,sharp);model.model.layers[li].mlp=w;s.mods[li]=w
        for p in model.parameters():p.requires_grad_(False)
    def _ids(s,t):return s.tok(t,return_tensors='pt',truncation=True,max_length=64).input_ids.to(s.model.device)
    def _centroid(s,texts):
        acc={li:[] for li in s.layers}
        for t in texts:
            with torch.no_grad():s.model(s._ids(t))
            for li in s.layers:acc[li].append(s.mods[li].last.float().mean(1).squeeze(0))
        return {li:torch.stack(acc[li],0).mean(0) for li in s.layers}
    def _ensure_mu(s):
        s.mu=s._centroid(GENERIC) if s.mu is None else s.mu
    def add_domain(s,name,facts,steps=420,lr=3e-4):
        s._ensure_mu();kc=s._centroid(list(facts))
        idx={li:s.mods[li].add(F.normalize(kc[li]-s.mu[li],dim=0),s.mu[li],s.r) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        for li in s.layers:s.mods[li].force[idx[li]]=True
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0
        for it in range(steps):
            e=s._ids(facts[it%len(facts)]);loss=F.cross_entropy(s.model(e).logits[0,:-1],e[0,1:]);opt.zero_grad();loss.backward();opt.step();lf=float(loss)
        s.model.eval()
        for li in s.layers:s.mods[li].force[idx[li]]=False
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def set_domain(s,name,on):
        for li,j in s.domains.get(name,{}).items():s.mods[li].on[j]=on
    def gen(s,prompt,max_new_tokens=12):
        e=s._ids(prompt)
        with torch.no_grad():o=s.model.generate(input_ids=e,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=s.tok.eos_token_id,use_cache=True)
        return s.tok.decode(o[0,e.shape[1]:],skip_special_tokens=True)
