import torch,pytest
from amni.inference import qgroup_gemv as Q
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="needs a GPU")
@pytest.mark.parametrize("bits,gs",[(2,64),(3,64),(4,128),(8,128)])
@pytest.mark.parametrize("M",[1,5,40])
def test_kernel_matches_dequant(bits,gs,M):
    torch.manual_seed(0);out,inn=300,1024;W=torch.randn(out,inn,device="cuda");x=torch.randn(M,inn,device="cuda").half()
    q,a,c=Q.quantize(W,bits,gs);pl=Q.split(q,bits);assert (Q.unpack(pl,bits,inn)==q).all()
    D=Q.dequant(pl,a,c,bits,gs,inn);y=Q.qgemm(pl,a,c,x,bits,gs).float();r=x.float()@D.T
    assert torch.nn.functional.cosine_similarity(y.flatten(),r.flatten(),dim=0)>0.9999
def test_error_monotone_in_bits():
    torch.manual_seed(1);W=torch.randn(256,512,device="cuda");e={}
    for b,gs in ((2,64),(3,64),(4,64),(8,64)):
        q,a,c=Q.quantize(W,b,gs);D=Q.dequant(Q.split(q,b),a,c,b,gs,512);e[b]=((D-W).norm()/W.norm()).item()
    assert e[8]<e[4]<e[3]<e[2]
