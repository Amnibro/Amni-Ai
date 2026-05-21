#!/usr/bin/env python
"""Download curated Piper voices to ~/.amni/voices/ for high-fidelity TTS.

Usage:
  python scripts/download_voices.py             # all curated voices (~150MB)
  python scripts/download_voices.py --voice amy # one voice
  python scripts/download_voices.py --list      # just show what's available
"""
import sys,os,argparse,urllib.request,hashlib
from pathlib import Path
try:sys.stdout.reconfigure(encoding='utf-8')
except Exception:pass
BASE='https://huggingface.co/rhasspy/piper-voices/resolve/main'
VOICES={
    'amy':       {'path':'en/en_US/amy/medium','name':'en_US-amy-medium',    'desc':'warm female (US), expressive',  'size_mb':62},
    'ryan':      {'path':'en/en_US/ryan/high', 'name':'en_US-ryan-high',     'desc':'clear male (US), high quality', 'size_mb':114},
    'lessac':    {'path':'en/en_US/lessac/medium','name':'en_US-lessac-medium','desc':'neutral male (US), professional','size_mb':62},
    'libritts':  {'path':'en/en_US/libritts_r/medium','name':'en_US-libritts_r-medium','desc':'multi-speaker (US), varied','size_mb':62},
    'alan':      {'path':'en/en_GB/alan/medium','name':'en_GB-alan-medium',  'desc':'British male, deep',            'size_mb':62},
    'jenny':     {'path':'en/en_GB/jenny_dioco/medium','name':'en_GB-jenny_dioco-medium','desc':'British female, friendly','size_mb':62},
}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--voice',default=None,help='download just this voice (key from --list)')
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--root',default=str(Path.home()/'.amni'/'voices'))
    args=ap.parse_args()
    if args.list:
        print('Available voices:')
        for k,v in VOICES.items():print(f'  {k:10s} {v["name"]:35s} {v["desc"]} ({v["size_mb"]} MB)')
        return
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    keys=[args.voice] if args.voice else list(VOICES.keys())
    if args.voice and args.voice not in VOICES:print(f'unknown voice: {args.voice} (try --list)',file=sys.stderr);sys.exit(2)
    for key in keys:
        v=VOICES[key];name=v['name']
        onnx_url=f'{BASE}/{v["path"]}/{name}.onnx'
        json_url=f'{BASE}/{v["path"]}/{name}.onnx.json'
        onnx_path=root/f'{name}.onnx';json_path=root/f'{name}.onnx.json'
        for url,dest in ((json_url,json_path),(onnx_url,onnx_path)):
            if dest.exists() and dest.stat().st_size>1024:
                print(f'  skip (exists): {dest.name}')
                continue
            print(f'  fetching {dest.name} ({v["size_mb"]} MB if onnx)... ',end='',flush=True)
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'AdamVoiceDownloader/1.0'})
                with urllib.request.urlopen(req,timeout=180) as r:
                    data=r.read();dest.write_bytes(data)
                print(f'OK ({len(data)//1024} KB)')
            except Exception as e:print(f'FAIL: {type(e).__name__}: {e}');continue
    print(f'\nVoices in {root}:')
    for f in sorted(root.glob('*.onnx')):print(f'  {f.name} ({f.stat().st_size//(1024*1024)} MB)')
if __name__=='__main__':main()
