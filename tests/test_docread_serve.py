import os,json,base64,shutil,tempfile
from pathlib import Path
import pytest
from amni.serve import docread as dr
FX=Path(__file__).resolve().parents[1]/'tools'/'docread'/'tests'/'fixtures'
pytestmark=pytest.mark.skipif(dr.binary() is None,reason='docread binary not built')
@pytest.fixture()
def store(monkeypatch):
    d=tempfile.mkdtemp(prefix='docread-test-');monkeypatch.setenv('AMNI_DOC_STORE',d);yield Path(d);shutil.rmtree(d,ignore_errors=True)
def test_info_pdf():
    i=dr.info(FX/'garden-lo.pdf');assert i['ok'] and i['format']=='pdf' and i['unit']=='page' and i['count']>=1
def test_attach_ask_cite_clear(store):
    r=dr.attach('',FX.joinpath('garden-lo.pdf').name,(FX/'garden-lo.pdf').read_bytes());sid=r['session_id']
    assert 'error' not in r and r['name']=='garden-lo.pdf' and len(dr.files(sid))==1
    h=dr.ask(sid,'when is the tool shed open on Saturdays?',2)
    assert h and h[0]['file']=='garden-lo.pdf' and h[0]['unit']=='page' and '8:00' in h[0]['text']
    f=dr.facts(sid,'What goes in the left compost bin?')
    assert f and f[0].startswith('Excerpt from the attached document (garden-lo.pdf, page') and 'Green waste' in f[0]
    assert dr.facts(sid,'what is the capital of peru')==[]
    assert dr.passages(sid,'summarize the document') and dr.clear(sid)['removed']==['garden-lo.pdf'] and dr.files(sid)==[]
def test_encrypted_and_formats(store):
    sid='t1'
    N=('garden-aes256.pdf','garden-lo-rc4.pdf','layout-aes256-objstm.pdf','sample.docx','sample.xlsx','sample.pptx','page.html','memo.rtf','notes.md','quoted.csv','sample.doc','legacy.xls','legacy.ppt','mail.eml','pk_ustar.tar.gz','pk_garden.txt.bz2','nb4.ipynb','cal.ics','cards.vcf','img_exif.png','img_sign.svg','code_src/hello.py')
    for n in N:
        r=dr.attach(sid,n,(FX/n).read_bytes());assert 'error' not in r,(n,r)
    names={Path(n).name for n in N};assert len(dr.listing(sid))==len(N)
    for q in ('compost','kite over a green hill','meeting','attachment budget','watering schedule'):
        h=dr.ask(sid,q,4);assert h and all(x['file'] in names for x in h),(q,[x.get('file') for x in h])
    for n,fm in (('sample.doc','ole'),('mail.eml','eml'),('pk_ustar.tar.gz','gzip'),('nb4.ipynb','ipynb'),('cal.ics','ics'),('cards.vcf','vcf'),('img_exif.png','image'),('code_src/hello.py','code')):
        i=dr.info(FX/n);assert i['ok'] and i['format']==fm and i['chars']>0,(n,i)
def test_rejects(store):
    assert 'error' in dr.attach('s',('x'*3)+'.bin',bytes(range(256))*64)
    assert dr.attach_path('s',str(FX/'notes.md'))=={'error':'path outside allowed roots'}
    assert dr.attach_path('s',str(FX/'notes.md'),lambda p:True)['name']=='notes.md'
    assert dr._name('../../etc/passwd')=='passwd' and dr._name('')=='document'
def test_harness_doc_route(store,monkeypatch):
    from amni.serve import adam_t_harness as H
    r=dr.attach('h1','garden-lo.pdf',(FX/'garden-lo.pdf').read_bytes())
    o=H._doc('What goes in the left compost bin?','h1',None,400)
    assert o and o['slots'][0]['kind']=='doc' and 'Green waste' in o['text'] and '(garden-lo.pdf, page' in o['text']
    assert H._doc('what is my name?','h1',None,400) is None and H._doc('compost','nobody',None,400) is None
def test_ingest_file_skill(store):
    from amni.serve.ingest import _ingest_file,_read_document
    class R:
        def _in_allowed_roots(s,p):return True
        def _abs(s,p):return Path(p)
    o=_ingest_file({'path':str(FX/'sample.docx'),'dry_run':True},{},R())
    assert o['reader']=='docread' and o['chunks_extracted']>=1 and o['source_kind']=='docx'
    d=_read_document({'path':str(FX/'garden-lo.pdf'),'question':'watering schedule on Wednesday'},{},R())
    assert d['passages'] and d['facts'][0].startswith('Excerpt from the attached document')
@pytest.fixture()
def fake_ocr(monkeypatch):
    seen=[];monkeypatch.setattr(dr,'ocr_cmd',lambda:'/bin/true');monkeypatch.setattr(dr,'ocr_langs',lambda:'eng');monkeypatch.setattr(dr,'caption',lambda d:'a printed receipt on a wooden table')
    monkeypatch.setattr(dr,'ocr',lambda p:seen.append(str(p)) or 'GREENLEAF NURSERY\nSeed trays 3 x 4.50\nTotal due 13.50 by Friday');yield seen
def test_image_ocr_sidecar_and_src(store,fake_ocr):
    r=dr.attach('o1','receipt.png',(FX/'img_text.png').read_bytes());assert 'error' not in r and r['ocr_words']>=9 and r['caption'] and r['info']['ok'] and r['info']['format']=='image'
    assert dr.listing('o1')==[{'name':'receipt.png','bytes':(FX/'img_text.png').stat().st_size,'derived_text':True}]
    h=dr.ask('o1','what is the total due?',2);assert h and h[0]['file']=='receipt.png' and h[0]['via']=='ocr' and h[0]['loc0']==0 and '13.50' in h[0]['text']
    f=dr.facts('o1','total due');assert f and 'text read from the image' in f[0] and '(receipt.png,' in f[0]
    assert dr.clear('o1')['removed']==['receipt.png'] and not list(dr._ddir('o1').glob('*.md'))
def test_src_strips_file_prefix():
    al=[Path('/s/a.pdf'),Path('/s/b.docx'),Path('/s/.derived/c.png.md')]
    assert dr._src({'head':'b.docx > Budget','text':'x'},al)=={'head':'Budget','text':'x','file':'b.docx'}
    o=dr._src({'head':'c.png.md > Text in c.png','unit':'section','loc0':2,'loc1':3,'text':'y'},al);assert o['file']=='c.png' and o['head']=='Text in c.png' and o['via']=='ocr' and (o['unit'],o['loc0'],o['loc1'])==('',0,0)
    assert dr._src({'head':'Intro','text':'z'},[Path('/s/a.pdf')])['file']=='a.pdf'
def test_scanned_pdf_pages_get_ocr(store,fake_ocr):
    d=dr._sdir('p1');d.mkdir(parents=True);dst=d/'scan.pdf';dst.write_bytes((FX/'pimg_rgb.pdf').read_bytes());i=dr.info(dst)
    assert i['ok'] and i['count']==1 and any('scanned' in w for w in i['warnings'])
    dr._scan('p1',dst,i['count']);t=(dr._ddir('p1')/'scan.pdf.md').read_text()
    assert t.startswith('# Scanned pages of scan.pdf (OCR)') and '# Page 1' in t and 'Total due 13.50' in t and len(fake_ocr)==1 and not Path(fake_ocr[0]).exists()
