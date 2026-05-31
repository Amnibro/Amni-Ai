"""One-shot upgrade for an existing Adam (Amni-Ai) git checkout.
Pulls the latest code (fast-forward), then re-runs the idempotent installer so Python deps
(`pip install -e .[all]` + requirements.txt), the amni_kernels native extension, and the
GF(17) bake (`amni init`) all come up to date in a single command. KaTeX (js/css/fonts) is
vendored + committed, so it arrives with the `git pull` - no separate step.

Usage:
  python upgrade.py                 # pull + update deps/kernels/model, then stop (you restart Adam)
  python upgrade.py --check         # DRY RUN: show exactly what would happen, change nothing
  python upgrade.py --launch        # pull + update, then launch the server
  python upgrade.py --skip-model    # pull + update deps only (keep current bake; e.g. no Gemma->Granite swap)
  python upgrade.py --gpu nvidia    # any install.py flag is passed straight through

Not a git checkout (downloaded a zip)? Clone fresh instead - your data lives in ~/.amni-ai, so it survives:
  git clone https://github.com/Amnibro/Amni-Ai && cd Amni-Ai && python install.py
"""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(cmd):
    print('  $ '+' '.join(cmd),flush=True)
    return subprocess.run(cmd).returncode
def _detect_gpu():
    try:
        import importlib.util as _il
        spec=_il.spec_from_file_location('_adam_install',str(ROOT/'install.py'));m=_il.module_from_spec(spec);spec.loader.exec_module(m)
        return m.detect_gpu('auto')
    except Exception as e:return 'detect-skipped ('+str(e)[:60]+')'
def check():
    venv=ROOT/'.venv';pyexe=(venv/('Scripts/python.exe' if sys.platform.startswith('win') else 'bin/python'))
    fwd=[a for a in sys.argv[1:] if a not in ('--check','--launch')]
    inst_args=fwd if '--launch' in sys.argv[1:] else ['--no-launch']+fwd
    print('=== DRY RUN - nothing will be changed ===',flush=True)
    if not (ROOT/'.git').exists():
        print('  git checkout: NO - a fresh clone is required:',flush=True)
        print('    git clone https://github.com/Amnibro/Amni-Ai && cd Amni-Ai && python install.py',flush=True);return 2
    print('  git checkout: yes - '+str(ROOT),flush=True)
    print('  --- current branch / local changes (git status -sb) ---',flush=True)
    subprocess.run(['git','-C',str(ROOT),'status','-sb'])
    print('  --- what the remote has (git fetch --dry-run, read-only) ---',flush=True)
    subprocess.run(['git','-C',str(ROOT),'fetch','--dry-run'])
    print('  venv: '+('exists -> reused' if venv.exists() else 'MISSING -> would be created'),flush=True)
    print('  GPU vendor (auto-detected, read-only): '+_detect_gpu(),flush=True)
    print('  amni_kernels imports for this Python: '+('yes -> no rebuild' if (pyexe.exists() and subprocess.run([str(pyexe),'-c','import amni_kernels'],capture_output=True).returncode==0) else 'no -> would build (Rust, one-time)'),flush=True)
    print('\n  WOULD RUN, in order (re-run without --check to apply):',flush=True)
    print('    1) git pull --ff-only            # code + KaTeX (js/css/fonts, committed) - no separate asset step',flush=True)
    print('    2) python install.py '+' '.join(inst_args),flush=True)
    print('         -> reuse venv, vendor-correct torch, pip install -e .[all] + requirements.txt,',flush=True)
    print('            build amni_kernels if needed, amni init (Granite GF(17) bake; multi-GB if switching from Gemma),',flush=True)
    print('            '+('then LAUNCH the server' if '--launch' in sys.argv[1:] else 'then STOP (you restart Adam)'),flush=True)
    print('\n  No files were modified. Add --skip-model to keep the current bake.',flush=True)
    return 0
def main():
    if '--check' in sys.argv[1:]:sys.exit(check())
    if not (ROOT/'.git').exists():
        print('[upgrade] This folder is not a git checkout, so there is nothing to `git pull`.',flush=True)
        print('  Get the latest with a fresh clone (your lessons/conversations in ~/.amni-ai survive):',flush=True)
        print('    git clone https://github.com/Amnibro/Amni-Ai && cd Amni-Ai && python install.py',flush=True)
        sys.exit(2)
    print('=== 1/2 git pull (fast-forward only) ===',flush=True)
    if run(['git','-C',str(ROOT),'pull','--ff-only'])!=0:
        print('[upgrade] git pull failed - likely local edits or a diverged branch.',flush=True)
        print('  Stash your changes and retry:  git stash && python upgrade.py   (git stash pop to restore)',flush=True)
        sys.exit(1)
    print('=== 2/2 re-running the installer (deps + kernels + model; KaTeX arrived with the pull) ===',flush=True)
    passthrough=[a for a in sys.argv[1:] if a!='--launch']
    args=passthrough if '--launch' in sys.argv[1:] else ['--no-launch']+passthrough
    rc=run([sys.executable,str(ROOT/'install.py')]+args)
    if rc==0:
        print('\n[upgrade] Done. '+('Adam is launching.' if '--launch' in sys.argv[1:] else 'Restart Adam to load the new version (e.g. `python -m amni.cli serve` or the adam launcher).'),flush=True)
    sys.exit(rc)
if __name__=='__main__':main()
