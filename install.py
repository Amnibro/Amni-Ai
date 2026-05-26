"""One-shot bootstrap installer for Adam.
Fully automated: detects Python, creates venv, detects GPU vendor, installs vendor-correct PyTorch, installs Adam, builds amni_kernels native extension (auto-installs Rust toolchain if missing), runs `amni init` (downloads bake), opens browser.
Usage:
  python install.py                                  # default: full auto install + launch
  python install.py --gpu nvidia                     # force NVIDIA CUDA torch
  python install.py --gpu amd                        # force AMD ROCm torch (Linux) / TheRock guidance (Windows)
  python install.py --gpu cpu                        # force CPU-only torch (small download, ~1 tok/s)
  python install.py --cuda cu121                     # override CUDA tag (default: cu124)
  python install.py --rocm rocm6.1                   # override ROCm tag (default: rocm6.2)
  python install.py --install-rust skip              # don't auto-install Rust toolchain even if amni_kernels needs it
  python install.py --skip-kernels                   # don't build amni_kernels (Adam runs with degraded paths)
  python install.py --no-launch                      # set up but don't start server
  python install.py --skip-model                     # skip ~20 GB bake download
  python install.py --bake-dir E:/Adam/bake          # store bake on external drive
  python install.py --home E:/Adam                   # store EVERYTHING on external drive
  python install.py --port 8080 --persona mentor     # custom port + persona
"""
import os,sys,subprocess,argparse,platform,shutil
from pathlib import Path
try:from amni.bootstrap import DEFAULT_PORT
except Exception:DEFAULT_PORT=7700
ROOT=Path(__file__).resolve().parent
VENV=ROOT/'.venv'
TORCH_PKGS=['torch','torchvision','torchaudio']
RUSTUP_WIN='https://win.rustup.rs/x86_64'
RUSTUP_UNIX='https://sh.rustup.rs'
def py():
    return str(VENV/'Scripts'/'python.exe') if platform.system()=='Windows' else str(VENV/'bin'/'python')
def run(cmd,env=None,**kw):
    print(f'  $ {" ".join(cmd) if isinstance(cmd,list) else cmd}',flush=True)
    r=subprocess.run(cmd,shell=isinstance(cmd,str),env=env,**kw)
    if r.returncode!=0:print(f'  [FAIL] exit {r.returncode}',flush=True);sys.exit(r.returncode)
    return r
def run_soft(cmd,env=None,**kw):
    print(f'  $ {" ".join(cmd) if isinstance(cmd,list) else cmd}',flush=True)
    return subprocess.run(cmd,shell=isinstance(cmd,str),env=env,**kw)
def step(s):print(f'\n=== {s} ===',flush=True)
def detect_gpu(override='auto'):
    if override!='auto':return override
    sys_os=platform.system()
    nvsmi=shutil.which('nvidia-smi')
    if nvsmi:
        try:
            r=subprocess.run([nvsmi,'--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
            if r.returncode==0 and r.stdout.strip():return 'nvidia'
        except Exception:pass
    if sys_os=='Linux' and shutil.which('rocminfo'):
        try:
            r=subprocess.run(['rocminfo'],capture_output=True,text=True,timeout=5)
            if r.returncode==0 and 'AMD' in r.stdout:return 'amd'
        except Exception:pass
    if sys_os=='Windows':
        ps=shutil.which('powershell') or shutil.which('pwsh')
        if ps:
            try:
                r=subprocess.run([ps,'-NoProfile','-Command','(Get-CimInstance Win32_VideoController).Name'],capture_output=True,text=True,timeout=10)
                out=(r.stdout or '').lower()
                if any(k in out for k in ('nvidia','geforce','rtx ','gtx ','quadro','tesla')):return 'nvidia'
                if any(k in out for k in ('radeon','amd ',' rx ','firepro')):return 'amd'
            except Exception:pass
    return 'cpu'
def torch_flavor(env):
    try:
        r=subprocess.run([py(),'-c','import torch;import sys;sys.stdout.write(torch.__version__);sys.stdout.write("|");sys.stdout.write(str(bool(torch.version.cuda)));sys.stdout.write("|");sys.stdout.write(getattr(torch.version,"hip","") or "")'],capture_output=True,text=True,timeout=30,env=env)
        if r.returncode!=0:return None
        parts=r.stdout.strip().split('|')
        return {'version':parts[0],'cuda':parts[1]=='True','hip':bool(parts[2])} if len(parts)==3 else None
    except Exception:return None
def torch_index(vendor,cuda_tag,rocm_tag):
    s=platform.system()
    if vendor=='nvidia':return f'https://download.pytorch.org/whl/{cuda_tag}'
    if vendor=='amd' and s=='Linux':return f'https://download.pytorch.org/whl/{rocm_tag}'
    if vendor=='cpu':return 'https://download.pytorch.org/whl/cpu'
    return None
def needs_reinstall(vendor,flavor):
    if flavor is None:return True
    if vendor=='nvidia':return not flavor['cuda']
    if vendor=='amd':return not flavor['hip']
    if vendor=='cpu':return flavor['cuda'] or flavor['hip']
    return False
def install_torch(vendor,cuda_tag,rocm_tag,env):
    s=platform.system()
    effective=vendor
    if vendor=='amd' and s=='Windows':
        print('  [WARN] AMD GPU on Windows: PyTorch.org has no official ROCm wheel.',flush=True)
        print('         Falling back to CPU torch. For AMD-on-Windows GPU acceleration see:',flush=True)
        print('           https://github.com/ROCm/TheRock  (community/experimental nightly builds)',flush=True)
        effective='cpu'
    flavor=torch_flavor(env)
    if flavor:print(f'  current torch: {flavor["version"]}  cuda={flavor["cuda"]}  hip={flavor["hip"] or "no"}',flush=True)
    if not needs_reinstall(effective,flavor):
        print(f'  torch flavor already matches {effective}; skipping reinstall',flush=True)
        return
    if flavor is not None:
        print(f'  reinstalling torch for {effective} (current does not match)',flush=True)
        subprocess.run([py(),'-m','pip','uninstall','-y']+TORCH_PKGS,env=env,capture_output=True)
    idx=torch_index(effective,cuda_tag,rocm_tag)
    args=[py(),'-m','pip','install','--upgrade']
    if idx:args+=['--index-url',idx]
    args+=TORCH_PKGS
    run(args,env=env)
def cargo_path():
    p=shutil.which('cargo')
    if p:return p
    cb=Path.home()/'.cargo'/'bin'/('cargo.exe' if platform.system()=='Windows' else 'cargo')
    return str(cb) if cb.exists() else None
def install_rust(env):
    s=platform.system()
    if s=='Windows':
        import urllib.request,tempfile
        tmp=Path(tempfile.gettempdir())/'rustup-init.exe'
        print(f'  downloading rustup-init.exe from {RUSTUP_WIN}  (~10 MB)',flush=True)
        urllib.request.urlretrieve(RUSTUP_WIN,str(tmp))
        run([str(tmp),'-y','--default-toolchain','stable','--profile','minimal','--no-modify-path'],env=env)
        try:tmp.unlink()
        except Exception:pass
    else:
        print(f'  downloading rustup install script from {RUSTUP_UNIX}',flush=True)
        run(f"curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_UNIX} | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path",env=env)
def amni_kernels_imports(env):
    import tempfile
    return subprocess.run([py(),'-c','import amni_kernels'],capture_output=True,env=env,cwd=tempfile.gettempdir()).returncode==0
def ensure_amni_kernels(env,install_rust_mode='auto',skip=False):
    if amni_kernels_imports(env):
        print('  amni_kernels native extension imports cleanly for this Python — skipping rebuild',flush=True)
        return True
    if skip:
        print('  [skip] --skip-kernels: amni_kernels will not be available (Adam runs with degraded paths)',flush=True)
        return False
    py_ver=f'{sys.version_info.major}.{sys.version_info.minor}'
    print(f'  amni_kernels prebuilt .pyd does not match running Python ({py_ver}) — building from source via maturin',flush=True)
    print('  this is a one-time ~5-15 min Rust compile + link step',flush=True)
    cargo=cargo_path()
    if not cargo:
        if install_rust_mode=='skip':
            print('  [WARN] Rust toolchain not found and --install-rust=skip; skipping amni_kernels build',flush=True)
            print('         Install Rust manually then re-run: https://rustup.rs/',flush=True)
            return False
        print('  [auto-install] Rust toolchain not found — installing minimal stable via rustup',flush=True)
        if platform.system()=='Windows':print('  [note] Windows Rust install needs MSVC build tools (Visual Studio Build Tools 2019+). rustup-init will prompt to install them if missing.',flush=True)
        try:install_rust(env)
        except SystemExit:
            print(f'  [WARN] Rust install failed. Install manually from https://rustup.rs/ then re-run `python install.py`',flush=True)
            return False
        except Exception as e:
            print(f'  [WARN] Rust install failed: {e}',flush=True)
            print(f'         Install manually from https://rustup.rs/ then re-run `python install.py`',flush=True)
            return False
        cargo=cargo_path()
        if not cargo:
            print('  [WARN] Rust appears installed but cargo not found on PATH or in ~/.cargo/bin; aborting amni_kernels build',flush=True)
            return False
    cargo_dir=str(Path(cargo).parent)
    env=dict(env);env['PATH']=cargo_dir+os.pathsep+env.get('PATH','')
    run([py(),'-m','pip','install','maturin>=1.12,<2.0'],env=env)
    r=run_soft([py(),'-m','maturin','develop','--release'],env=env,cwd=str(ROOT/'amni_kernels'))
    if r.returncode!=0:
        print(f'  [WARN] maturin build exited {r.returncode}; proceeding with degraded paths',flush=True)
        return False
    if amni_kernels_imports(env):
        print('  amni_kernels built + installed for this Python ABI',flush=True)
        return True
    print('  [WARN] amni_kernels build completed but import still fails — proceeding with degraded paths',flush=True)
    return False
def main():
    ap=argparse.ArgumentParser(description='Adam one-shot installer',formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)
    ap.add_argument('--no-launch',action='store_true',help='Set up but do not start the server')
    ap.add_argument('--skip-model',action='store_true',help='Skip model download (~20 GB Gemma-4 bake)')
    ap.add_argument('--bake-dir',default=None,help='Custom path for the bake (default: <home>/bakes/gemma4_e2b_it_gf17)')
    ap.add_argument('--home',default=None,help='Custom config home for EVERYTHING — bakes, lessons, conversations (default: ~/.amni-ai)')
    ap.add_argument('--port',type=int,default=DEFAULT_PORT)
    ap.add_argument('--persona',default='alfred',help='Default persona (alfred=loyal-butler default; override with rikku/yoda/mentor/etc)')
    ap.add_argument('--gpu',choices=['auto','nvidia','amd','cpu'],default='auto',help='GPU vendor for PyTorch (default: auto-detect)')
    ap.add_argument('--cuda',default='cu124',help='CUDA tag for NVIDIA torch (default: cu124)')
    ap.add_argument('--rocm',default='rocm6.2',help='ROCm tag for AMD-Linux torch (default: rocm6.2)')
    ap.add_argument('--install-rust',choices=['auto','skip'],default='auto',help='Auto-install Rust via rustup when amni_kernels needs to be built (default: auto)')
    ap.add_argument('--skip-kernels',action='store_true',help='Skip building amni_kernels native extension entirely (Adam runs with degraded paths)')
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
    print(f'Working dir: {Path.cwd()}',flush=True)
    if sys.version_info<(3,10):print('  [FAIL] Python 3.10+ required.');sys.exit(1)
    if not (ROOT/'pyproject.toml').exists() or not (ROOT/'amni').exists():
        print(f'\n[FAIL] install.py is not inside an Amni-Ai checkout.',flush=True)
        print(f'  Expected pyproject.toml + amni/ next to install.py at: {ROOT}',flush=True)
        print(f'  Likely cause: `git clone` failed (DNS, network, or firewall) and you ran install.py from the wrong directory.',flush=True)
        print(f'  Fix: cd to where you want Adam, then:',flush=True)
        print(f'    git clone https://github.com/Amnibro/Amni-Ai',flush=True)
        print(f'    cd Amni-Ai',flush=True)
        print(f'    python install.py',flush=True)
        sys.exit(2)
    step('1/6 Creating virtual environment')
    if VENV.exists():print(f'  venv exists at {VENV} — reusing.',flush=True)
    else:run([sys.executable,'-m','venv',str(VENV)])
    run([py(),'-m','pip','install','--upgrade','pip','wheel'],env=env)
    step('2/6 Detecting GPU and installing vendor-correct PyTorch')
    vendor=detect_gpu(args.gpu)
    print(f'  GPU vendor: {vendor}  (override with --gpu nvidia|amd|cpu)',flush=True)
    install_torch(vendor,args.cuda,args.rocm,env)
    step('3/6 Installing Adam')
    run([py(),'-m','pip','install','-e','.[all]'],env=env)
    step('4/6 Building amni_kernels native extension (if needed for this Python)')
    ensure_amni_kernels(env,install_rust_mode=args.install_rust,skip=args.skip_kernels)
    step('5/6 Initializing Adam (config + lessons + optional model download)')
    init_cmd=[py(),'-m','amni.cli','init','--non-interactive']
    if args.skip_model:init_cmd.append('--skip-model')
    run(init_cmd,env=env)
    step('6/6 Launching Adam')
    if args.no_launch:
        print(f'  Setup complete. Run: {py()} -m amni.cli serve --port {args.port} --default-persona {args.persona}',flush=True)
        return
    serve_cmd=[py(),'-m','amni.cli','serve','--port',str(args.port),'--default-persona',args.persona,'--open-browser']
    print(f'  Adam will open in your browser at http://127.0.0.1:{args.port}/',flush=True)
    run(serve_cmd,env=env)
if __name__=='__main__':main()
