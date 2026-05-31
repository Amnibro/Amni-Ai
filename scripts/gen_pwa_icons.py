"""Generate Adam PWA icons (maskable): a glowing cyan 'A' on near-black. Run: python scripts/gen_pwa_icons.py"""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFilter
OUT=Path(__file__).resolve().parents[1]/'amni'/'serve'/'assets'/'icons'
OUT.mkdir(parents=True,exist_ok=True)
BG=(4,7,17,255);CYAN=(0,229,255,255)
def _lerp(a,b,t):return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
def make(sz):
    img=Image.new('RGBA',(sz,sz),BG)
    layer=Image.new('RGBA',(sz,sz),(0,0,0,0));d=ImageDraw.Draw(layer)
    m=sz*0.26;w=max(2,int(sz*0.072))
    apex=(sz/2,m);left=(m,sz-m);right=(sz-m,sz-m)
    d.line([apex,left],fill=CYAN,width=w);d.line([apex,right],fill=CYAN,width=w)
    t=0.62;cbl=_lerp(apex,left,t);cbr=_lerp(apex,right,t)
    d.line([cbl,cbr],fill=CYAN,width=int(w*0.82))
    glow=layer.filter(ImageFilter.GaussianBlur(sz*0.025))
    img=Image.alpha_composite(img,glow);img=Image.alpha_composite(img,glow);img=Image.alpha_composite(img,layer)
    return img
for s in (192,512):
    p=OUT/f'adam-{s}.png';make(s).save(p,'PNG')
    print('wrote',p,p.stat().st_size,'bytes')
