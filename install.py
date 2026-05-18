"""One-shot bootstrap installer for Adam.
Cross-platform: detects Python, creates venv, installs deps, downloads model, runs `amni init`, opens browser.
Usage:
  python install.py                                  # default install + launch
  python install.py --no-launch                      # set up but don't start server
  python install.py --skip-model                     # skip ~5 GB bake download
  python install.py --bake-dir E:/Adam/bake          # store bake on external drive
  python install.py --home E:/Adam                   # store EVERYTHING (config, bake, lessons, conversations) on external drive
  python install.py --port 8080 --persona mentor     # custom port + persona
"""
import os,sys,subprocess,argparse,platform,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VENV=ROOT/'.venv'
def py():
    return str(VENV/'Scripts'/'python.exe') if platform.system()=='Windows' else str(VENV/'bin'/'python')
def run(cmd,env=None,**kw):
    print(f'  $ {" ".join(cmd) if isinstance(cmd,list) else cmd}',flush=True)
    r=subprocess.run(cmd,shell=isinstance(cmd,str),env=env,**kw)
    if r.returncode!=0:print(f'  [FAIL] exit {r.returncode}',flush=True);sys.exit(r.returncode)
    return r
def step(s):print(f'\n=== {s} ===',flush=True)
def main():
    ap=argparse.ArgumentParser(description='Adam one-shot installer',formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)
    ap.add_argument('--no-launch',action='store_true',help='Set up but do not start the server')
    ap.add_argument('--skip-model',action='store_true',help='Skip model download (~5 GB Gemma-4 bake)')
    ap.add_argument('--bake-dir',default=None,help='Custom path for the bake (default: <home>/bakes/gemma4_e2b_it_gf17)')
    ap.add_argument('--home',default=None,help='Custom config home for EVERYTHING — bakes, lessons, conversations (default: ~/.amni-ai)')
    ap.add_argument('--port',type=int,default=8002)
    ap.add_argument('--persona',default='rikku')
    args=ap.parse_args()
    env=os.environ.copy()
    if args.home:
        h=Path(args.home).expanduser().resolve();h.mkdir(parents=True,exist_ok=True)
        env['AMNI_HOME']=str(h);print(f'  AMNI_HOME set: {h}',flush=True)
    if args.bake_dir:
        b=Path(args.bake_dir).expanduser().resolve();b.mkdir(parents=True,exist_ok=True)
        env['AMNI_BAKE']=str(b);print(f'  AMNI_BAKE set: {b}',flush=True)
    print('Adam — Amni-Ai one-shot installer',flush=True)
    print(f'Platform: {platform.system()} {platform.release()} | Python: {sys.version.split()[0]}',flush=True)
    if sys.version_info<(3,10):print('  [FAIL] Python 3.10+ required.');sys.exit(1)
    step('1/4 Creating virtual environment')
    if VENV.exists():print(f'  venv exists at {VENV} — reusing.',flush=True)
    else:run([sys.executable,'-m','venv',str(VENV)])
    step('2/4 Installing Adam')
    run([py(),'-m','pip','install','--upgrade','pip','wheel'],env=env)
    run([py(),'-m','pip','install','-e','.[all]'],env=env)
    step('3/4 Initializing Adam (config + lessons + optional model download)')
    init_cmd=[py(),'-m','amni.cli','init','--non-interactive']
    if args.skip_model:init_cmd.append('--skip-model')
    run(init_cmd,env=env)
    step('4/4 Launching Adam')
    if args.no_launch:
        print(f'  Setup complete. Run: {py()} -m amni.cli serve --port {args.port} --default-persona {args.persona}',flush=True)
        return
    serve_cmd=[py(),'-m','amni.cli','serve','--port',str(args.port),'--default-persona',args.persona,'--open-browser']
    print(f'  Adam will open in your browser at http://127.0.0.1:{args.port}/',flush=True)
    run(serve_cmd,env=env)
if __name__=='__main__':main()
