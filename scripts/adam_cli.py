#!/usr/bin/env python
"""Adam CLI — terminal client for a running local Adam (default: http://127.0.0.1:11434).
Usage:
  python scripts/adam_cli.py "your prompt"
  python scripts/adam_cli.py --session work1 "follow-up"
  python scripts/adam_cli.py --persona Yoda "explain recursion"
  python scripts/adam_cli.py --port 11434 --host 127.0.0.1 "hi"
  echo "hi" | python scripts/adam_cli.py --stdin
Flags: --json (raw events), --no-color, --max-time 180, --skill calc (direct skill invoke)
"""
import sys,os,argparse,json,time
try:import requests
except ImportError:print('pip install requests',file=sys.stderr);sys.exit(2)
def _color(s,c):return s if os.environ.get('NO_COLOR') else f'\x1b[{c}m{s}\x1b[0m'
def _green(s):return _color(s,'32')
def _dim(s):return _color(s,'2')
def _bold(s):return _color(s,'1')
def _cyan(s):return _color(s,'36')
def _yellow(s):return _color(s,'33')
def _red(s):return _color(s,'31')
def _record_wav(seconds:float,samplerate:int=16000)->bytes:
    import sounddevice as sd,wave,io,numpy as np
    print(_yellow(f'[recording {seconds}s — speak now]'),file=sys.stderr)
    audio=sd.rec(int(seconds*samplerate),samplerate=samplerate,channels=1,dtype='int16');sd.wait()
    buf=io.BytesIO()
    with wave.open(buf,'wb') as wf:wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(samplerate);wf.writeframes(audio.tobytes())
    return buf.getvalue()
def _play_wav(audio_bytes:bytes):
    import sounddevice as sd,wave,io,numpy as np
    buf=io.BytesIO(audio_bytes)
    with wave.open(buf,'rb') as wf:
        sr=wf.getframerate();frames=wf.readframes(wf.getnframes())
        arr=np.frombuffer(frames,dtype=np.int16)
        if wf.getnchannels()==2:arr=arr.reshape(-1,2)
    sd.play(arr,sr);sd.wait()
def main():
    ap=argparse.ArgumentParser(description='Adam CLI — talk to a local Adam server')
    ap.add_argument('prompt',nargs='*',help='Prompt text (or use --stdin / --voice-record)')
    ap.add_argument('--host',default=os.environ.get('ADAM_HOST','127.0.0.1'))
    ap.add_argument('--port',type=int,default=int(os.environ.get('ADAM_PORT','11434')))
    ap.add_argument('--session',default=None,help='Session ID (continuity across runs)')
    ap.add_argument('--persona',default=None,help='Persona name (Rikku/Yoda/Mentor/etc)')
    ap.add_argument('--max-time',type=float,default=300.0,help='Network timeout seconds')
    ap.add_argument('--json',action='store_true',help='Output raw SSE events as JSON lines')
    ap.add_argument('--no-color',action='store_true')
    ap.add_argument('--stdin',action='store_true',help='Read prompt from stdin')
    ap.add_argument('--skill',default=None,help='Invoke a skill directly instead of chat (e.g. --skill calc)')
    ap.add_argument('--no-stream',action='store_true',help='Wait for full response then print')
    ap.add_argument('--voice-record',type=float,default=0,metavar='SECONDS',help='Record N seconds of mic and send via /voice/chat')
    ap.add_argument('--play',action='store_true',help='Play response audio through speakers')
    ap.add_argument('--continuous',action='store_true',help='With --voice-record, loop until Ctrl-C')
    args=ap.parse_args()
    if args.no_color:os.environ['NO_COLOR']='1'
    base=f'http://{args.host}:{args.port}'
    if args.voice_record>0:
        import base64 as _b64
        while True:
            try:
                wav=_record_wav(args.voice_record)
                payload={'audio_base64':_b64.b64encode(wav).decode('ascii'),'return_audio':True}
                if args.session:payload['session_id']=args.session
                try:r=requests.post(f'{base}/voice/chat',json=payload,timeout=args.max_time)
                except Exception as e:print(_red(f'connection: {e}'),file=sys.stderr);sys.exit(1)
                if r.status_code!=200:print(_red(f'HTTP {r.status_code}: {r.text[:300]}'),file=sys.stderr);sys.exit(1)
                d=r.json()
                if d.get('stt_error'):print(_red(f'STT: {d["stt_error"]}'),file=sys.stderr)
                if d.get('transcript'):print(_dim('[you] ')+d['transcript'])
                print(_green(f'[{d.get("persona","Adam")}] ')+d.get('response',''))
                if d.get('audio_base64'):
                    try:_play_wav(_b64.b64decode(d['audio_base64']))
                    except Exception as e:print(_red(f'play error: {e}'),file=sys.stderr)
                print(_dim(f'[tier={d.get("tier","?")} wall={d.get("wall_s","?")}s session={d.get("session_id","?")}]'),file=sys.stderr)
                if not args.continuous:break
            except KeyboardInterrupt:print(_dim('\n[exiting voice loop]'),file=sys.stderr);break
        return
    msg=' '.join(args.prompt).strip() if args.prompt else (sys.stdin.read().strip() if args.stdin else '')
    if not msg:print('Usage: adam_cli "your prompt here"  |  --voice-record 5  |  --stdin',file=sys.stderr);sys.exit(2)
    if args.persona:
        try:requests.post(f'{base}/persona',json={'name':args.persona},timeout=10).raise_for_status()
        except Exception as e:print(_red(f'[persona-switch failed: {e}]'),file=sys.stderr)
    if args.skill:
        try:
            r=requests.post(f'{base}/skills/{args.skill}',json={'args':{'cmd':msg} if args.skill in ('shell','git') else ({'expr':msg} if args.skill=='calc' else ({'query':msg} if args.skill in ('mem','web') else {'path':msg}))},timeout=args.max_time)
            print(json.dumps(r.json(),indent=2))
        except Exception as e:print(_red(f'skill error: {e}'),file=sys.stderr);sys.exit(1)
        return
    payload={'message':msg}
    if args.session:payload['session_id']=args.session
    try:resp=requests.post(f'{base}/chat/stream',json=payload,stream=True,timeout=args.max_time)
    except Exception as e:print(_red(f'connection error: {e}'),file=sys.stderr);sys.exit(1)
    if resp.status_code!=200:
        print(_red(f'HTTP {resp.status_code}: {resp.text[:300]}'),file=sys.stderr);sys.exit(1)
    t0=time.time();ev='token';data='';sid='';first=True;persona_name=''
    if args.no_stream:
        full=[]
        for line in resp.iter_lines(decode_unicode=True):
            if not line:continue
            if line.startswith('event:'):ev=line[6:].strip()
            elif line.startswith('data:'):
                d=line[5:].strip()
                if ev=='token':
                    try:full.append(json.loads(d))
                    except:full.append(d)
                elif ev=='meta':
                    try:m=json.loads(d);sid=m.get('session_id',sid);persona_name=m.get('persona',persona_name)
                    except:pass
        print(''.join(full))
        print(_dim(f'\n[session={sid} persona={persona_name} wall={round(time.time()-t0,1)}s]'),file=sys.stderr)
        return
    for line in resp.iter_lines(decode_unicode=True):
        if not line:continue
        if line.startswith('event:'):ev=line[6:].strip()
        elif line.startswith('data:'):
            d=line[5:].strip()
            if args.json:print(json.dumps({'event':ev,'data':d}));continue
            if ev=='token':
                try:tok=json.loads(d);sys.stdout.write(tok);sys.stdout.flush()
                except:pass
            elif ev=='meta':
                try:
                    m=json.loads(d);sid=m.get('session_id',sid);persona_name=m.get('persona',persona_name)
                    if first and persona_name:print(_cyan(f'[{persona_name}] ')+_dim('thinking...'),end='\r',file=sys.stderr);first=False
                    if m.get('skill'):print(_yellow(f'\r[skill: {m["skill"]}]'),file=sys.stderr)
                    if m.get('agentic'):print(_yellow('\r[agentic mode]'),file=sys.stderr)
                except:pass
            elif ev=='thinking':
                try:t=json.loads(d);print(_dim(f'\r[thinking {t.get("elapsed",0)}s, buffering {t.get("buf_chars",0)} chars]   '),end='',file=sys.stderr)
                except:pass
            elif ev.startswith('agentic_'):
                try:t=json.loads(d);phase=t.get('event','');tool=t.get('tool','');print(_yellow(f'\n[{phase}{(" -> "+tool) if tool else ""}]'),file=sys.stderr)
                except:pass
            elif ev=='web_lookup':print(_yellow('\n[web lookup...]'),file=sys.stderr)
            elif ev=='done':
                try:dd=json.loads(d);print(_dim(f'\n\n[tier={dd.get("tier","?")} wall={dd.get("wall_s","?")}s session={sid}]'),file=sys.stderr)
                except:pass
            elif ev=='error':print(_red(f'\n[error: {d}]'),file=sys.stderr)
    print('')
if __name__=='__main__':main()
