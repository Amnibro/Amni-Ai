"""Gated density pages — runtime-addressed weight learning (the proven densify_gated recipe, productionized).
Each learning = a low-rank residual page on late-MLP layers + a Reffelt-nonce address key (mean centered residual-stream hidden of its prompts).
gate=sigmoid((cos(x-mu,key)-TAU)*SHARP) -> the page speaks ONLY when a query routes near its address: plasticity on-domain, zero collateral off-domain, additive across independent pages, exactly removable. Train gate-open (strong memorize), serve sharp-gated."""
import torch,torch.nn as nn,torch.nn.functional as F
GENERIC=["The capital of France is Paris.","Water is made of hydrogen and oxygen.","Two plus two equals four.","The sun rises in the east.","A dog is a kind of animal.","The novel explores themes of love and loss.","Photosynthesis converts sunlight into energy.","Newton described the laws of motion.","The weather today is sunny and warm.","DNA carries genetic information.","Shakespeare wrote many famous plays.","The ocean covers most of the planet."]
class _GatedMLP(nn.Module):
    def __init__(s,mlp,hs,tau,sharp):
        super().__init__();s.base=mlp;s.hs=hs;s.tau=tau;s.sharp=sharp;s.pg=nn.ParameterList();s.keys=[];s.mus=[];s.on=[];s.force=[];s.lg=[];s.slot=[];s.gw=[];s.gb=[];s.hard=False;s.router=False;s.floor=None;s.last=None
        for p in s.base.parameters():p.requires_grad_(False)
    def add(s,key,muv,r,slot=None,gw=None,gb=None):
        A=nn.Parameter(torch.zeros(r,s.hs,device=key.device,dtype=torch.float32));B=nn.Parameter(torch.randn(s.hs,r,device=key.device,dtype=torch.float32)*1e-3)
        s.pg.append(A);s.pg.append(B);s.keys.append(key);s.mus.append(muv);s.on.append(True);s.force.append(False);s.lg.append(0.0);s.slot.append(slot);s.gw.append(gw);s.gb.append(gb);return len(s.keys)-1
    def _cos(s,xf,j):return F.cosine_similarity(xf-s.mus[j],s.keys[j].expand_as(xf),dim=-1)
    def forward(s,x):
        s.last=x.detach();y=s.base(x);xf=x.float();rj=[j for j in range(len(s.keys)) if s.on[j] and not s.force[j]];win=None
        if s.router and rj:
            S=torch.stack([s._cos(xf,j) for j in rj],-1);win=S.argmax(-1);win=torch.where(S.max(-1).values>=s.floor,win,torch.full_like(win,-1)) if s.floor is not None else win
        for j in range(len(s.keys)):
            if s.on[j]:
                c=(xf@s.pg[2*j].t())@s.pg[2*j+1].t()
                if s.slot[j] is not None:c=c*s.slot[j]
                if s.force[j]:y=y+c.to(y.dtype)
                else:
                    g=((win==rj.index(j)).float() if win is not None else ((s._cos(xf,j)>s.tau).float() if s.hard else torch.sigmoid((s._cos(xf,j)-s.tau)*s.sharp))).unsqueeze(-1);s.lg[j]=float(g.mean());y=y+(g*c).to(y.dtype)
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
    def add_domain_supervised(s,name,prompts,targets,steps=900,lr=3e-4,maxlen=448):
        if s.mu is None:s.mu=s._centroid(GENERIC)
        acc={li:[] for li in s.layers}
        for p in prompts[:200]:
            with torch.no_grad():s.model(s.tok(p,return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device))
            for li in s.layers:acc[li].append(s.mods[li].last.float().mean(1).squeeze(0))
        kc={li:torch.stack(acc[li],0).mean(0) for li in s.layers}
        idx={li:s.mods[li].add(F.normalize(kc[li]-s.mu[li],dim=0),s.mu[li],s.r) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        for li in s.layers:s.mods[li].force[idx[li]]=True
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0;n=len(prompts)
        for it in range(steps):
            j=it%n;e=s.tok(prompts[j],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device);tgt=torch.tensor([int(targets[j])],device=s.model.device)
            loss=F.cross_entropy(s.model(e).logits[0,-1:],tgt);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();lf=0.98*lf+0.02*float(loss) if it else float(loss)
        s.model.eval()
        for li in s.layers:s.mods[li].force[idx[li]]=False
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def add_domain_keyed(s,name,prompts,targets,keys,muv,steps=1800,lr=1e-4,maxlen=448):
        idx={li:s.mods[li].add(keys[li],muv[li],s.r) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        for li in s.layers:s.mods[li].force[idx[li]]=True
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0;n=len(prompts)
        for it in range(steps):
            j=it%n;e=s.tok(prompts[j],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device);tgt=torch.tensor([int(targets[j])],device=s.model.device)
            loss=F.cross_entropy(s.model(e).logits[0,-1:],tgt);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();lf=0.98*lf+0.02*float(loss) if it else float(loss)
        s.model.eval()
        for li in s.layers:s.mods[li].force[idx[li]]=False
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def add_domain_corrective(s,name,prompts,corr,wrong,keys,muv,steps=1200,lr=1e-4,margin=4.0,maxlen=448):
        idx={li:s.mods[li].add(keys[li],muv[li],s.r) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        for li in s.layers:s.mods[li].force[idx[li]]=True
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0;n=len(prompts)
        for it in range(steps):
            j=it%n;e=s.tok(prompts[j],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device);lg=s.model(e).logits[0,-1]
            loss=F.relu(margin-(lg[int(corr[j])]-lg[int(wrong[j])]));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();lf=0.98*lf+0.02*float(loss) if it else float(loss)
        s.model.eval()
        for li in s.layers:s.mods[li].force[idx[li]]=False
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def add_domain_disjoint(s,name,prompts,targets,keys,muv,slot,steps=1800,lr=1e-4,maxlen=448):
        idx={li:s.mods[li].add(keys[li],muv[li],s.r,slot) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        for li in s.layers:s.mods[li].force[idx[li]]=True
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0;n=len(prompts)
        for it in range(steps):
            j=it%n;e=s.tok(prompts[j],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device);tgt=torch.tensor([int(targets[j])],device=s.model.device)
            loss=F.cross_entropy(s.model(e).logits[0,-1:],tgt);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();lf=0.98*lf+0.02*float(loss) if it else float(loss)
        s.model.eval()
        for li in s.layers:s.mods[li].force[idx[li]]=False
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def add_domain_silenced(s,name,prompts,targets,offprompts,keys,muv,steps=1400,lr=1e-4,lam=3.0,maxlen=448):
        idx={li:s.mods[li].add(keys[li],muv[li],s.r) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0;n=len(prompts);no=len(offprompts)
        for it in range(steps):
            for li in s.layers:s.mods[li].force[idx[li]]=True
            j=it%n;e=s.tok(prompts[j],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device);tgt=torch.tensor([int(targets[j])],device=s.model.device)
            la=F.cross_entropy(s.model(e).logits[0,-1:],tgt)
            for li in s.layers:s.mods[li].force[idx[li]]=False
            k=it%no;eo=s.tok(offprompts[k],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device)
            for li in s.layers:s.mods[li].on[idx[li]]=False
            with torch.no_grad():base=s.model(eo).logits[0,-1].detach()
            for li in s.layers:s.mods[li].on[idx[li]]=True
            ls=F.mse_loss(s.model(eo).logits[0,-1],base)
            loss=la+lam*ls;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();lf=0.98*lf+0.02*float(la) if it else float(la)
        s.model.eval()
        for li in s.layers:s.mods[li].force[idx[li]]=False
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def add_domain_trained_gate(s,name,prompts,targets,offprompts,keys,muv,steps=1500,lr=1e-4,gsteps=400,glr=0.05,maxlen=448):
        def feats(plist):
            acc={li:[] for li in s.layers}
            for p in plist:
                with torch.no_grad():s.model(s.tok(p,return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device))
                for li in s.layers:acc[li].append(s.mods[li].last.float().mean(1).squeeze(0))
            return {li:torch.stack(acc[li],0) for li in s.layers}
        pos=feats(prompts[:130]);neg=feats(offprompts[:130]);gw={};gb={}
        for li in s.layers:
            X=torch.cat([pos[li],neg[li]],0);Y=torch.cat([torch.ones(pos[li].shape[0]),torch.zeros(neg[li].shape[0])]).to(X.device)
            w=torch.zeros(X.shape[1],device=X.device,requires_grad=True);b=torch.zeros(1,device=X.device,requires_grad=True);go=torch.optim.Adam([w,b],lr=glr,weight_decay=2e-2)
            for _ in range(gsteps):
                l=F.binary_cross_entropy_with_logits(X@w+b,Y);go.zero_grad();l.backward();go.step()
            gw[li]=w.detach();gb[li]=b.detach()
        idx={li:s.mods[li].add(keys[li],muv[li],s.r,None,gw[li],gb[li]) for li in s.layers}
        ps=[s.mods[li].pg[2*idx[li]+i] for li in s.layers for i in (0,1)]
        for p in ps:p.requires_grad_(True)
        opt=torch.optim.Adam(ps,lr=lr);s.model.train();lf=0.0;n=len(prompts)
        for it in range(steps):
            j=it%n;e=s.tok(prompts[j],return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device);tgt=torch.tensor([int(targets[j])],device=s.model.device)
            loss=F.cross_entropy(s.model(e).logits[0,-1:],tgt);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ps,1.0);opt.step();lf=0.98*lf+0.02*float(loss) if it else float(loss)
        s.model.eval()
        for p in ps:p.requires_grad_(False)
        s.domains[name]=idx;return lf
    def gate_rate(s,name,prompts,maxlen=448):
        idx=s.domains[name];tot=0.0
        for p in prompts:
            with torch.no_grad():s.model(s.tok(p,return_tensors='pt',truncation=True,max_length=maxlen).input_ids.to(s.model.device))
            tot+=sum(s.mods[li].lg[idx[li]] for li in s.layers)/len(s.layers)
        return tot/max(1,len(prompts))
    def set_domain(s,name,on):
        for li,j in s.domains.get(name,{}).items():
            if li in s.mods:s.mods[li].on[j]=on
    def save(s,path):
        st={'layers':s.layers,'r':s.r,'domains':s.domains,'pages':{li:[(s.mods[li].pg[2*j].detach().cpu(),s.mods[li].pg[2*j+1].detach().cpu(),s.mods[li].keys[j].cpu(),s.mods[li].mus[j].cpu(),(s.mods[li].slot[j].cpu() if s.mods[li].slot[j] is not None else None)) for j in range(len(s.mods[li].keys))] for li in s.layers}}
        torch.save(st,path);return path
    def load(s,path):
        st=torch.load(path,map_location='cpu');dev=s.model.device;s.domains=st['domains']
        for li in st['layers']:
            if li not in s.mods:continue
            for A,B,key,mu,slot in st['pages'][li]:
                idx=s.mods[li].add(key.to(dev),mu.to(dev),s.r,(slot.to(dev) if slot is not None else None))
                with torch.no_grad():s.mods[li].pg[2*idx].copy_(A.to(dev));s.mods[li].pg[2*idx+1].copy_(B.to(dev))
        return len(s.domains)
    def gen(s,prompt,max_new_tokens=12):
        e=s._ids(prompt)
        with torch.no_grad():o=s.model.generate(input_ids=e,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=s.tok.eos_token_id,use_cache=True)
        return s.tok.decode(o[0,e.shape[1]:],skip_special_tokens=True)
