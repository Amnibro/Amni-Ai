"""kg_query — relational queries on Adam's KnowledgeGraph. Skill surface.
Actions:
  neighbors <subject>      → all outgoing+incoming edges
  out <subject>            → outgoing edges only
  in <subject>             → incoming edges only
  predicate <p>            → all triples with that predicate
  path <a> <b>             → BFS shortest path (max_hops, default 3)
  search <q>               → fuzzy subject search
  add s,p,o                → manual triple add
  forget [s|p|o]           → bulk forget by filter
  stats                    → graph stats"""
from typing import Dict,Any,List,Optional
def _widget(title:str,data:Dict[str,Any])->Dict[str,Any]:return {'type':'info','title':title,'icon':'🧠','data':data}
def kg_query_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    kg=ctx.get('knowledge_graph') if ctx else None
    if kg is None:return {'error':'KnowledgeGraph not in skill context'}
    action=(args.get('action') or '').strip().lower()
    if action in ('stats','status',''):
        s=kg.stats();return {**s,'widget':_widget('Knowledge Graph stats',s)}
    if action in ('neighbors','neighbours'):
        subj=(args.get('subject') or args.get('q') or '').strip()
        if not subj:return {'error':'need subject'}
        edges=kg.neighbors_of(subj,direction=args.get('direction','both'),limit=int(args.get('limit',30)))
        return {'subject':subj,'edges':edges,'n':len(edges),'widget':_widget(f'Neighbors · {subj}',{'subject':subj,'edges_n':len(edges),'sample':edges[:6]})}
    if action=='out':
        return kg_query_skill({**args,'action':'neighbors','direction':'out'},ctx,reg)
    if action=='in':
        return kg_query_skill({**args,'action':'neighbors','direction':'in'},ctx,reg)
    if action in ('predicate','by_predicate'):
        p=(args.get('predicate') or args.get('p') or '').strip()
        if not p:return {'error':'need predicate'}
        out=kg.by_predicate(p,limit=int(args.get('limit',30)))
        return {'predicate':p,'triples':out,'n':len(out),'widget':_widget(f'Predicate · {p}',{'predicate':p,'triples_n':len(out),'sample':out[:6]})}
    if action=='path':
        a=(args.get('a') or args.get('from') or '').strip();b=(args.get('b') or args.get('to') or '').strip()
        if not a or not b:return {'error':'need from + to'}
        max_hops=int(args.get('max_hops',3))
        p=kg.path_between(a,b,max_hops=max_hops)
        if p is None:return {'from':a,'to':b,'path':None,'reason':f'no path within {max_hops} hops','widget':_widget(f'Path · {a} → {b}',{'from':a,'to':b,'found':False,'max_hops':max_hops})}
        if not p:return {'from':a,'to':b,'path':[],'reason':'same subject','widget':_widget(f'Path · {a} → {b}',{'from':a,'to':b,'found':True,'length':0})}
        return {'from':a,'to':b,'path':p,'hops':len(p),'widget':_widget(f'Path · {a} → {b}',{'from':a,'to':b,'found':True,'length':len(p),'edges':[(e['s'],e['p'],e['o']) for e in p]})}
    if action=='search':
        q=(args.get('q') or args.get('subject') or '').strip()
        if not q:return {'error':'need q'}
        hits=kg.search_subject(q,limit=int(args.get('limit',20)))
        return {'q':q,'matches':hits,'n':len(hits)}
    if action=='add':
        s=(args.get('s') or args.get('subject') or '').strip()
        p=(args.get('p') or args.get('predicate') or '').strip()
        o=(args.get('o') or args.get('object') or '').strip()
        if not (s and p and o):return {'error':'need s, p, o'}
        added=kg.add(s,p,o,source=args.get('source','manual'),confidence=float(args.get('confidence',0.8) or 0.8),kind='manual')
        return {'added':added is not None,'triple':added}
    if action=='forget':
        n=kg.forget(subject=args.get('subject'),predicate=args.get('predicate'),object_=args.get('object'))
        return {'forgot':n}
    return {'error':f'unknown action "{action}"; valid: stats|neighbors|out|in|predicate|path|search|add|forget'}
