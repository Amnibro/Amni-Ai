"""Adam-1 quickstart: load bake, infer, log experience, distill, observe overlay.

Run after `python scripts/adam1_bake.py --hf-id Qwen/Qwen2.5-1.5B-Instruct --out bakes/qwen25_1_5b_gf17`.

Usage:
    python examples/quickstart.py --bake bakes/qwen25_1_5b_gf17 --model downloaded_models/.../Qwen2.5-1.5B-Instruct
"""
import argparse,sys,shutil
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
import torch
from amni.inference.streaming_chat import StreamingChatService
from amni.learning import LearningWriter,ResidualSFTLearner,ExperienceAtlas,PrismTexBundle
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bake',required=True)
    ap.add_argument('--model',required=True,help='HF source model path (provides tokenizer/config)')
    ap.add_argument('--budget-mb',type=int,default=8000)
    ap.add_argument('--workdir',default='examples/quickstart_workdir')
    ap.add_argument('--skip-train',action='store_true',help='skip the SFT step (just demo storage+inference)')
    args=ap.parse_args()
    workdir=Path(args.workdir)
    workdir.mkdir(parents=True,exist_ok=True)
    print('='*60);print('Adam-1 quickstart');print('='*60)
    print()
    print('[1] Load bake + classify foundational tiers')
    w=LearningWriter(args.bake)
    if w.tier_summary().get('wisdom',0)==0:w.assign_tiers()
    summary=w.tier_summary()
    total=sum(summary.values())
    locked=total-summary.get('wisdom',0)
    print(f'    {total} tensors total, {locked} tier-locked, {summary.get("wisdom",0)} writable (wisdom)')
    print(f'    breakdown: {dict(summary)}')
    print()
    print('[2] Boot StreamingChatService (8 GB VRAM cache)')
    svc=StreamingChatService(args.bake,args.model,budget_mb=args.budget_mb,enable_prefetch=True)
    if hasattr(svc.registry,'set_active_subjects'):svc.registry.set_active_subjects(['global'])
    print(f'    bake loaded; active_subjects=[global]')
    print()
    print('[3] Run inference (baseline, no residuals)')
    questions=[
        ('What is 12 + 7?','math'),
        ('Write a Python one-liner to compute the factorial of 5.','code'),
        ('Name three primary colors.','general'),
    ]
    print('    [pre-training inferences]')
    pre_responses=[]
    for q,_ in questions:
        r,_=svc.chat(q,max_new_tokens=64,do_sample=False)
        pre_responses.append(r)
        print(f'      Q: {q}')
        print(f'      A: {r[:140]}')
    print()
    if args.skip_train:
        print('[4] --skip-train set, demonstrating storage primitives only')
    else:
        print('[4] Log a few experiences to a PTEX texture-map atlas')
        atlas_root=workdir/'atlas'
        if atlas_root.exists():shutil.rmtree(atlas_root)
        atlas=ExperienceAtlas(atlas_root,subject='quickstart-demo')
        for q,resp in zip([q for q,_ in questions],pre_responses):
            atlas.append(q,resp,outcome=1,system='You are a helpful assistant.')
        for i in range(50):
            atlas.append(f'What is {i}*2?',f'The answer is {i*2}.',outcome=1,system='Answer math problems concisely.')
        s=atlas.stats()
        print(f'    atlas now has {s["n_records"]} records, {s["total_bytes"]:,} bytes')
        print()
        print('[5] Distill atlas into Wisdom-tier residuals (Asimov/Foundation tiers untouched)')
        del svc
        learner=ResidualSFTLearner(args.bake,args.model,trainable_layer_min=20,verbose=True)
        learner.load_model()
        train_stats,n_encoded=learner.train_from_atlas(atlas,subject='quickstart-demo',epochs=1,batch_size=1,grad_accum=8,lr=2e-5,max_len=384)
        learner.shutdown()
        print(f'    distilled {n_encoded} tensors under subject="quickstart-demo", final_loss={train_stats["final_avg_loss"]:.4f}')
        print()
        print('[6] Re-boot inference + activate the distilled subject (must match the subject the residuals were tagged with)')
        svc=StreamingChatService(args.bake,args.model,budget_mb=args.budget_mb,enable_prefetch=True)
        svc.registry.set_active_subjects(['quickstart-demo'])
        print(f'    [post-training inferences with subjects=["quickstart-demo"] active]')
        for q,_ in questions:
            r,_=svc.chat(q,max_new_tokens=64,do_sample=False)
            print(f'      Q: {q}')
            print(f'      A: {r[:140]}')
        print()
        print('[6b] Same queries, subject="auto" — SubjectClassifier picks per-query (v5.5.54)')
        print('     Note: classifier picks math/code/etc. Our trained residual is under "quickstart-demo"')
        print('     not those subjects, so auto routing here demonstrates classification, not residual application.')
        for q,expected in questions:
            r,_=svc.chat(q,subject='auto',max_new_tokens=64,do_sample=False)
            picked=','.join(svc.registry.active_subjects)
            print(f'      Q: {q}')
            print(f'      [auto -> subject={picked}] A: {r[:120]}')
    print()
    print('[7] Export the residuals as a subject-tagged PrismTex bundle for federation')
    w.reload()
    has_residuals=False
    for s in w.list_subjects():
        if w.list_residual_tensors(subject=s):has_residuals=True;break
    if has_residuals:
        bundle=PrismTexBundle.export_from_bake(args.bake,contributor_id='quickstart-demo',subject='quickstart-demo',note='from quickstart')
        out=workdir/'shareable.prismtex'
        bundle.write(out)
        print(f'    exported {out} ({out.stat().st_size:,} bytes)')
        print(f'    bundle.header.subject={bundle.header.get("subject")}')
        print(f'    -> for cross-Adam federation: PrismTexBundle.merge_fp16_avg([this, peer1, peer2, ...], base_bake)')
        print(f'    -> for single-bundle apply: bundle.apply_to_bake(target_bake) writes to that subject is residual file')
    else:
        print('    (no residuals to export)')
    print()
    print('[8] Roll back: clear all residuals (returns Adam to immutable base)')
    cleared=0
    w.reload()
    for s in w.list_subjects():
        for k in w.list_residual_tensors(subject=s):
            w.clear_residuals(k,subject=s)
            cleared+=1
    print(f'    cleared {cleared} residual tensors; base weights unchanged on disk')
    print()
    print('='*60)
    print('QUICKSTART DONE')
    print('Storage:    GF(17) digit planes, bit-exact.')
    print('Tiers:      Asimov+Commandments+Ascension+Foundation never modified (142 tensors locked).')
    print('Wisdom:     residuals applied via (d_base + l) mod 17, reversible.')
    print('Subjects:   each tensor can carry many subject-tagged residual files; per-subject routing is bit-perfect.')
    print('Routing:    svc.chat(q, subject="auto") picks the right overlay automatically (SubjectClassifier).')
    print('Federation: PrismTexBundle.merge_fp16_avg([bundleA, bundleB, ...], base_bake) for N-Adam consensus.')
    print('Multi-subj: subjects=[X,Y] active simultaneously uses fp16-avg overlay decode (no collapse).')
    print('='*60)
if __name__=='__main__':main()
