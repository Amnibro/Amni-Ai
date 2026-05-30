import numpy as np,torch,torch.nn.functional as F
def pack_topk(ids,logprobs):
    ids=np.asarray(ids,dtype=np.int32);lp=np.asarray(logprobs,dtype=np.float16)
    return np.array([ids.shape[0],ids.shape[1]],dtype=np.int32).tobytes()+ids.tobytes()+lp.tobytes()
def unpack_topk(buf):
    t,k=(int(x) for x in np.frombuffer(buf[:8],dtype=np.int32));o=8+t*k*4
    return np.frombuffer(buf[8:o],dtype=np.int32).reshape(t,k),np.frombuffer(buf[o:],dtype=np.float16).reshape(t,k)
def kl_topk_loss(student_logits,topk_ids,topk_logprobs,mask=None):
    s_lp=F.log_softmax(torch.gather(student_logits,-1,torch.as_tensor(topk_ids,device=student_logits.device,dtype=torch.long)),dim=-1)
    t_p=F.softmax(topk_logprobs.to(student_logits.dtype),dim=-1)
    kl=(t_p*(torch.log(t_p.clamp_min(1e-12))-s_lp)).sum(-1)
    return kl.mean() if mask is None else (kl*mask).sum()/mask.sum().clamp_min(1.0)
