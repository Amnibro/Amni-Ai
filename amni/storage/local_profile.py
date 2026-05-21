import json,time
from pathlib import Path
class LocalProfile:
    def __init__(self,path,fact_re):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.fact_re=fact_re;self.data=self._load()
    def _load(self):
        try:return json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        except Exception:return {}
    def save(self):
        try:self.path.write_text(json.dumps(self.data,indent=2,ensure_ascii=False),encoding='utf-8')
        except Exception as e:print(f'[LocalProfile] save failed: {e}',flush=True)
    def update_from_message(self,text):
        if not text or not self.fact_re:return False
        changed=False
        for m in self.fact_re.finditer(text):
            name=m.group(1);fav_t=m.group(2);fav_v=m.group(3);like=m.group(4);loc_live=m.group(5) if m.lastindex and m.lastindex>=5 else None;loc_from=m.group(6) if m.lastindex and m.lastindex>=6 else None;workplace=m.group(7) if m.lastindex and m.lastindex>=7 else None;role=m.group(8) if m.lastindex and m.lastindex>=8 else None;title=m.group(9) if m.lastindex and m.lastindex>=9 else None
            if name:
                n=name.strip()
                if self.data.get('name')!=n:self.data['name']=n;changed=True
            elif fav_t and fav_v:
                k=fav_t.strip().lower();v=fav_v.strip();self.data.setdefault('favorites',{})
                if self.data['favorites'].get(k)!=v:self.data['favorites'][k]=v;changed=True
            elif like:
                lk=like.strip();self.data.setdefault('likes',[])
                if lk not in self.data['likes']:self.data['likes'].append(lk);changed=True
            elif loc_live or loc_from:
                lv=(loc_live or loc_from).strip().rstrip('.,')
                if self.data.get('location')!=lv:self.data['location']=lv;changed=True
            elif workplace:
                wp=workplace.strip().rstrip('.,')
                if self.data.get('workplace_or_project')!=wp:self.data['workplace_or_project']=wp;changed=True
            elif role or title:
                rt=(role or title).strip().rstrip('.,')
                if self.data.get('role')!=rt:self.data['role']=rt;changed=True
        if changed:self.data['updated_ts']=time.time();self.save()
        return changed
    def to_facts_list(self):
        out=[]
        if self.data.get('name'):out.append(f"user's name is {self.data['name']}")
        if self.data.get('location'):out.append(f"user lives in / is located in {self.data['location']}")
        if self.data.get('role'):out.append(f"user's role/occupation is {self.data['role']}")
        if self.data.get('workplace_or_project'):out.append(f"user works at/on {self.data['workplace_or_project']}")
        for k,v in (self.data.get('favorites') or {}).items():out.append(f"user's favorite {k} is {v}")
        for lk in (self.data.get('likes') or []):out.append(f"user likes/prefers {lk}")
        return out
    def forget(self):self.data={};self.save()
    def stats(self):return {'has_name':bool(self.data.get('name')),'has_location':bool(self.data.get('location')),'has_role':bool(self.data.get('role')),'has_workplace':bool(self.data.get('workplace_or_project')),'favorites_n':len(self.data.get('favorites') or {}),'likes_n':len(self.data.get('likes') or []),'updated_ts':self.data.get('updated_ts')}
