"""Expose Adam to your phone over a Cloudflare Tunnel, gated by an access token.

What it does:
  1. Makes (or reuses) a persistent access token at ~/.amni-ai/mobile_token.txt
  2. (optional) launches the Adam server itself with that token via --serve
  3. runs `cloudflared` to publish http://localhost:<port> at a public https URL
  4. prints a one-tap mobile link  https://<url>/?token=<token>  (+ a QR if `qrcode` is installed)

The token gate is enforced by the server (AMNI_AUTH_TOKEN): every API call needs it, so the
public URL is NOT an open door to Adam's shell/file/PC skills. The phone stores the token after
the first ?token= visit; "Add to Home Screen" then behaves like a native app.

Usage:
  python scripts/tunnel.py                 # assumes Adam is already running on :7700 with the token (see note it prints)
  python scripts/tunnel.py --serve         # also start Adam (with the token) for you
  python scripts/tunnel.py --port 7700 --serve --persona rikku
  python scripts/tunnel.py --new-token     # rotate the token
"""
import os,sys,re,subprocess,shutil,argparse,secrets,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HOME=Path(os.environ.get('AMNI_HOME') or (Path.home()/'.amni-ai'))
TOKEN_FILE=HOME/'mobile_token.txt'
URL_RE=re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')
def load_token(rotate=False):
    if not rotate:
        env=(os.environ.get('AMNI_AUTH_TOKEN') or '').strip()
        if env:return env
        if TOKEN_FILE.exists():
            t=TOKEN_FILE.read_text(encoding='utf-8').strip()
            if t:return t
    HOME.mkdir(parents=True,exist_ok=True)
    t=secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(t,encoding='utf-8')
    print(f'  [token] new access token written to {TOKEN_FILE}',flush=True)
    return t
def cloudflared_bin():
    return shutil.which('cloudflared')
def install_hint():
    s=sys.platform
    print('\n[cloudflared not found] install it (free, no account needed for quick tunnels):',flush=True)
    if s.startswith('win'):print('  winget install --id Cloudflare.cloudflared    (or: https://github.com/cloudflare/cloudflared/releases)',flush=True)
    elif s=='darwin':print('  brew install cloudflared',flush=True)
    else:print('  see https://pkg.cloudflare.com/  (apt/yum) or https://github.com/cloudflare/cloudflared/releases',flush=True)
def print_qr(url):
    try:
        import qrcode
        qr=qrcode.QRCode(border=1);qr.add_data(url);qr.make()
        qr.print_ascii(invert=True)
    except Exception:
        print('  (install `pip install qrcode` to show a scannable QR here)',flush=True)
def main():
    ap=argparse.ArgumentParser(description='Adam Cloudflare tunnel + token gate',formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)
    ap.add_argument('--port',type=int,default=int(os.environ.get('AMNI_PORT','7700')))
    ap.add_argument('--serve',action='store_true',help='also launch the Adam server with the token')
    ap.add_argument('--persona',default='alfred',help='persona when using --serve')
    ap.add_argument('--new-token',action='store_true',help='rotate the access token')
    args=ap.parse_args()
    token=load_token(rotate=args.new_token)
    print(f'  access token: {token}',flush=True)
    print(f'  token file  : {TOKEN_FILE}',flush=True)
    cf=cloudflared_bin()
    if not cf:install_hint();sys.exit(3)
    srv=None
    if args.serve:
        env=os.environ.copy();env['AMNI_AUTH_TOKEN']=token
        py=str((ROOT/'.venv'/('Scripts/python.exe' if sys.platform.startswith('win') else 'bin/python')))
        if not Path(py).exists():py=sys.executable
        print(f'  launching Adam (token-gated) on :{args.port} …',flush=True)
        srv=subprocess.Popen([py,'-m','amni.cli','serve','--port',str(args.port),'--cors','--default-persona',args.persona],cwd=str(ROOT),env=env)
    else:
        print('\n  NOTE: start Adam in another terminal with the SAME token, e.g.:',flush=True)
        if sys.platform.startswith('win'):print(f'    set AMNI_AUTH_TOKEN_FILE={TOKEN_FILE}\n    python -m amni.cli serve --port {args.port} --cors',flush=True)
        else:print(f'    AMNI_AUTH_TOKEN_FILE="{TOKEN_FILE}" python3 -m amni.cli serve --port {args.port} --cors',flush=True)
    print('\n  starting cloudflared … (Ctrl+C to stop)\n',flush=True)
    try:
        proc=subprocess.Popen([cf,'tunnel','--url',f'http://localhost:{args.port}'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        shown=False
        for line in proc.stdout:
            sys.stdout.write(line);sys.stdout.flush()
            if not shown:
                m=URL_RE.search(line)
                if m:
                    shown=True;link=m.group(0)+f'/?token={token}'
                    print('\n'+'='*64,flush=True)
                    print('  ADAM IS LIVE FOR YOUR PHONE — open this once, then Add to Home Screen:',flush=True)
                    print('   '+link,flush=True)
                    print('='*64,flush=True)
                    print_qr(link)
                    print('  (the ?token logs the phone in + is then stripped from the URL)\n',flush=True)
        proc.wait()
    except KeyboardInterrupt:print('\n  stopping tunnel…',flush=True)
    finally:
        if srv is not None:
            try:srv.terminate()
            except Exception:pass
if __name__=='__main__':main()
