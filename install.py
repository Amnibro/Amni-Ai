"""One-shot bootstrap installer for Adam.
Cross-platform Python script: detects Python, creates venv, installs deps, optionally downloads model, runs `amni init`, opens browser.
Usage:
  python install.py                  # full setup + launch
  python install.py --no-launch      # set up but don't start server
  python install.py --skip-model     # skip model download (BYO bake path)
"""
import os,sys,subprocess,argparse,platform,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VENV=ROOT/'.venv'
def py():
    return str(VENV/'Scripts'/'python.exe') if platform.system()=='Windows' else str(VENV/'bin'/'python')
def run(cmd,**kw):
    print(f'  $ {" ".join(cmd) if isinstance(cmd,list) else cmd}',flush=True)
    r=subprocess.run(cmd,shell=isinstance(cmd,str),**kw)
    if r.returncode!=0:print(f'  [FAIL] exit {r.returncode}',flush=True);sys.exit(r.returncode)
    return r
def step(s):print(f'\n=== {s} ===',flush=True)
def main():
    ap=argparse.ArgumentParser(description='Adam one-shot installer')
    ap.add_argument('--no-launch',action='store_true',help='Set up but do not start the server')
    ap.add_argument('--skip-model',action='store_true',help='Skip model download (configure manually later)')
    ap.add_argument('--port',type=int,default=8002)
    ap.add_argument('--persona',default='rikku')
    args=ap.parse_args()
    print('Adam — Amni-Ai one-shot installer',flush=True)
    print(f'Platform: {platform.system()} {platform.release()} | Python: {sys.version.split()[0]}',flush=True)
    if sys.version_info<(3,10):print('  [FAIL] Python 3.10+ required.');sys.exit(1)
    step('1/4 Creating virtual environment')
    if VENV.exists():print(f'  venv exists at {VENV} — reusing.',flush=True)
    else:run([sys.executable,'-m','venv',str(VENV)])
    step('2/4 Installing Adam')
    run([py(),'-m','pip','install','--upgrade','pip','wheel'])
    run([py(),'-m','pip','install','-e','.[all]'])
    step('3/4 Initializing Adam (config + lessons + optional model download)')
    init_cmd=[py(),'-m','amni.cli','init','--non-interactive']
    if args.skip_model:init_cmd.append('--skip-model')
    run(init_cmd)
    step('4/4 Launching Adam')
    if args.no_launch:
        print(f'  Setup complete. Run: {py()} -m amni.cli serve --port {args.port} --default-persona {args.persona}',flush=True)
        return
    serve_cmd=[py(),'-m','amni.cli','serve','--port',str(args.port),'--default-persona',args.persona,'--open-browser']
    print(f'  Adam will open in your browser at http://127.0.0.1:{args.port}/',flush=True)
    run(serve_cmd)
if __name__=='__main__':main()
