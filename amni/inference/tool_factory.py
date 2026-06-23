"""ToolFactory — turn Adam loose building its OWN toolkit (Anthony 2026-06-23). Runs Adam (via AdamCoder) over a queue of tool specs; each tool is generated -> sanitized -> VERIFIED (functional test or mocked system-call test, so NO real device/network touched during build) -> banked as macros -> written to adam_tools/<name>.py and registered. Only verified tools are saved. The registry is a growing capability set Adam can later call (each tool also becomes reusable macro blocks -> the ratchet accelerates future tool-building). Building is safe (mocked); USING a tool on real hardware is a separate gated action."""
import os,json
class ToolFactory:
    def __init__(s,coder,out_dir='adam_tools'):
        s.coder=coder;s.out=out_dir;os.makedirs(out_dir,exist_ok=True);s.registry={}
    def build(s,specs):
        results=[]
        for sp in specs:
            print(f'\n>>> Adam building tool: {sp["name"]} — {sp["desc"]}',flush=True)
            r=s.coder.write(sp['task'],test_fn=sp.get('test'),scaffold=sp.get('scaffold',''),max_tries=sp.get('tries',3),max_tokens=sp.get('tokens',700))
            if r['verified']:
                path=os.path.join(s.out,sp['name']+'.py')
                open(path,'w',encoding='utf-8').write(r['code'])
                s.registry[sp['name']]={'path':path,'desc':sp['desc'],'attempts':r['attempts']}
                print(f'    VERIFIED in {r["attempts"]} attempt(s) -> {path} (banked {r.get("banked_macros",0)} macros)',flush=True)
            else:
                print(f'    UNVERIFIED after {r["attempts"]} attempts: {r.get("error")}',flush=True)
            results.append({'name':sp['name'],'verified':r['verified'],'attempts':r['attempts'],'code':r['code']})
        json.dump({k:{'path':v['path'],'desc':v['desc']} for k,v in s.registry.items()},open(os.path.join(s.out,'registry.json'),'w'),indent=1)
        return results
