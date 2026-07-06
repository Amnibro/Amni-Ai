"""Amni-Chat Python client — wire-compatible with the Android app and the Rust server.
Envelope: X25519(ECDH) -> HKDF-SHA256(salt=32x00, info=b"amni-chat-v1", 32B) -> ChaCha20-Poly1305 (12B nonce, 16B tag, no AAD).
All addresses/keys/nonce/ciphertext are lowercase hex on the wire. Plaintext is the UTF-8 message body.
Crypto backend: prefers `cryptography`, falls back to `PyNaCl`; HKDF is stdlib. Transport: prefers `requests`, falls back to urllib."""
import json, os, hmac, hashlib, secrets, urllib.parse, urllib.request, re
INFO = b"amni-chat-v1"
_HKDF_SALT = b"\x00" * 32
def _hkdf_sha256(ikm, length=32, salt=_HKDF_SALT, info=INFO):
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest(); out += t; i += 1
    return out[:length]
def _load_backend():
    try:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        from cryptography.hazmat.primitives import serialization
        raw = serialization.Encoding.Raw; pf = serialization.PublicFormat.Raw; prf = serialization.PrivateFormat.Raw; ne = serialization.NoEncryption()
        def x_kp():
            k = X25519PrivateKey.generate(); return k.private_bytes(raw, prf, ne), k.public_key().public_bytes(raw, pf)
        def x_shared(priv, pub): return X25519PrivateKey.from_private_bytes(priv).exchange(X25519PublicKey.from_public_bytes(pub))
        def ed_kp():
            k = Ed25519PrivateKey.generate(); return k.private_bytes(raw, prf, ne), k.public_key().public_bytes(raw, pf)
        def aenc(key, nonce, pt): return ChaCha20Poly1305(key).encrypt(nonce, pt, None)
        def adec(key, nonce, ct): return ChaCha20Poly1305(key).decrypt(nonce, ct, None)
        return "cryptography", x_kp, x_shared, ed_kp, aenc, adec
    except Exception:
        from nacl import bindings as b
        from nacl.signing import SigningKey
        def x_kp():
            priv = secrets.token_bytes(32); return priv, b.crypto_scalarmult_base(priv)
        def x_shared(priv, pub): return b.crypto_scalarmult(priv, pub)
        def ed_kp():
            sk = SigningKey.generate(); return bytes(sk), bytes(sk.verify_key)
        def aenc(key, nonce, pt): return b.crypto_aead_chacha20poly1305_ietf_encrypt(pt, b"", nonce, key)
        def adec(key, nonce, ct): return b.crypto_aead_chacha20poly1305_ietf_decrypt(ct, b"", nonce, key)
        return "pynacl", x_kp, x_shared, ed_kp, aenc, adec
_BACKEND, _x_kp, _x_shared, _ed_kp, _aenc, _adec = _load_backend()
def crypto_backend(): return _BACKEND
def normalize(identifier):
    t = (identifier or "").strip().lower()
    return t if "@" in t else "".join(c for c in t if c.isdigit() or c == "+")
def hmac_id(pepper, principal):
    return hmac.new(pepper.encode() if isinstance(pepper, str) else pepper, normalize(principal).encode(), hashlib.sha256).hexdigest()
def fingerprint(ed_pub_bytes):
    h = ed_pub_bytes.hex()
    return " ".join(h[i:i + 4] for i in range(0, len(h), 4))
class Identity:
    def __init__(self, ed_priv, ed_pub, x_priv, x_pub):
        self.ed_priv, self.ed_pub, self.x_priv, self.x_pub = ed_priv, ed_pub, x_priv, x_pub
    @property
    def ed_hex(self): return self.ed_pub.hex()
    @property
    def x_hex(self): return self.x_pub.hex()
    @classmethod
    def generate(cls):
        edp, edpub = _ed_kp(); xp, xpub = _x_kp(); return cls(edp, edpub, xp, xpub)
    def to_dict(self): return {"ed_priv": self.ed_priv.hex(), "ed_pub": self.ed_pub.hex(), "x_priv": self.x_priv.hex(), "x_pub": self.x_pub.hex()}
    @classmethod
    def from_dict(cls, d): return cls(bytes.fromhex(d["ed_priv"]), bytes.fromhex(d["ed_pub"]), bytes.fromhex(d["x_priv"]), bytes.fromhex(d["x_pub"]))
    def save(self, path):
        with open(path, "w") as f: json.dump(self.to_dict(), f)
    @classmethod
    def load_or_create(cls, path):
        if os.path.exists(path):
            with open(path) as f: return cls.from_dict(json.load(f))
        idn = cls.generate(); idn.save(path); return idn
def seal(my_x_priv, their_x_pub, plaintext):
    key = _hkdf_sha256(_x_shared(my_x_priv, their_x_pub)); nonce = secrets.token_bytes(12); return nonce, _aenc(key, nonce, plaintext)
def unseal(my_x_priv, their_x_pub, nonce, ciphertext):
    key = _hkdf_sha256(_x_shared(my_x_priv, their_x_pub)); return _adec(key, nonce, ciphertext)
def parse_profile_link(raw):
    m = re.search(r"(amni://contact\?\S+|https?://\S+?/add\?\S+)", (raw or "").strip())
    if not m: return None
    q = urllib.parse.urlparse(m.group(1)).query; p = urllib.parse.parse_qs(q)
    ed = (p.get("ed") or [""])[0].lower(); x = (p.get("x") or [""])[0].lower()
    if len(ed) != 64 or len(x) != 64: return None
    return {"principal": (p.get("p") or [""])[0], "ed": ed, "x": x, "fp": (p.get("fp") or [""])[0], "nostr": (p.get("np") or [None])[0]}
def clean_wire_body(s):
    if not s: return s
    if s[0]=='\x1a':
        sep=s.find('\x1c',1);s=s[sep+1:] if sep>=0 else s
    if not s: return None
    c=s[0]
    if c=='\x01':
        rest=s[1:]
        if not rest.startswith('R:'): return s
        content=rest[2:];nl=content.find('\n')
        return content if nl<0 else content[nl+1:]
    return None if ord(c)<32 and c!='\n' else s

class AmniChatClient:
    def __init__(self, server_url, identity, pepper=None, timeout=20):
        self.base = server_url.rstrip("/"); self.id = identity; self.pepper = pepper; self.timeout = timeout
        try:
            import requests; self._req = requests.Session()
        except Exception:
            self._req = None
    def _post_json(self, path, obj):
        data = json.dumps(obj).encode()
        if self._req: return self._req.post(self.base + path, data=data, headers={"Content-Type": "application/json"}, timeout=self.timeout).json()
        r = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(r, timeout=self.timeout) as resp: return json.loads(resp.read().decode())
    def _get_json(self, path):
        if self._req: return self._req.get(self.base + path, timeout=self.timeout).json()
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as resp: return json.loads(resp.read().decode())
    def capabilities(self): return self._get_json("/capabilities")
    def register(self, principal=None, push_token="", platform="bot", nostr_pub=None, device_id=""):
        hid = hmac_id(self.pepper, principal) if (principal and self.pepper) else secrets.token_hex(32)
        body = {"hmac_id": hid, "ed25519_pub": self.id.ed_hex, "x25519_pub": self.id.x_hex, "push_token": push_token, "platform": platform, "device_id": device_id}
        if nostr_pub: body["nostr_pub"] = nostr_pub
        return self._post_json("/register", body)
    def lookup_by_principal(self, principal):
        if not self.pepper: raise ValueError("pepper required for principal lookup")
        return self._get_json("/lookup?hmac_id=" + hmac_id(self.pepper, principal))
    def lookup_by_ed(self, ed_hex): return self._get_json("/lookup/by-ed?ed=" + ed_hex)
    def send_text(self, to_ed_hex, to_x_hex, text):
        nonce, ct = seal(self.id.x_priv, bytes.fromhex(to_x_hex), text.encode("utf-8"))
        return self._post_json("/inbox/put", {"to": to_ed_hex.lower(), "from": self.id.ed_hex, "nonce": nonce.hex(), "ciphertext": ct.hex()})
    def blob_put(self, blob_id, enc):
        url = self.base + "/blob/put?id=" + blob_id
        if self._req: return self._req.post(url, data=enc, headers={"Content-Type": "application/octet-stream"}, timeout=self.timeout).status_code < 400
        r = urllib.request.Request(url, data=enc, headers={"Content-Type": "application/octet-stream"}, method="POST")
        with urllib.request.urlopen(r, timeout=self.timeout) as resp: return getattr(resp, "status", 200) < 400
    def blob_get(self, blob_id):
        url = self.base + "/blob/get?id=" + blob_id
        if self._req:
            r = self._req.get(url, timeout=self.timeout)
            return r.content if r.status_code < 400 else None
        with urllib.request.urlopen(url, timeout=self.timeout) as resp: return resp.read()
    def fetch_attachment(self, frame):
        if not frame or frame[0] != "\x04": return None
        parts = frame[1:].split(":", 4)
        if len(parts) < 5: return None
        blob_id, key_hex, nonce_hex, mime, name = parts
        if not re.fullmatch(r"[0-9a-fA-F-_]{2,64}", blob_id): return None
        enc = self.blob_get(blob_id)
        if not enc: return None
        try: data = _adec(bytes.fromhex(key_hex), bytes.fromhex(nonce_hex), enc)
        except Exception: return None
        return {"data": data, "mime": mime, "name": name}
    def send_attachment(self, to_ed_hex, to_x_hex, data, mime="image/png", name="chart.png"):
        blob_key = secrets.token_bytes(32); blob_nonce = secrets.token_bytes(12)
        enc = _aenc(blob_key, blob_nonce, data)
        blob_id = secrets.token_bytes(16).hex()
        if not self.blob_put(blob_id, enc): return None
        wire = "\x04" + blob_id + ":" + blob_key.hex() + ":" + blob_nonce.hex() + ":" + mime + ":" + name
        nonce, ct = seal(self.id.x_priv, bytes.fromhex(to_x_hex), wire.encode("utf-8"))
        return self._post_json("/inbox/put", {"to": to_ed_hex.lower(), "from": self.id.ed_hex, "nonce": nonce.hex(), "ciphertext": ct.hex()})
    def send_to_principal(self, principal, text):
        d = self.lookup_by_principal(principal); ed = d.get("ed25519_pub"); x = d.get("x25519_pub")
        if not ed or not x: raise LookupError("no device for principal")
        return self.send_text(ed, x, text)
    def drain(self,verbose=False):
        out = []
        for it in self._get_json("/inbox/drain?pubkey=" + self.id.ed_hex).get("items", []):
            from_ed = it.get("from", ""); from_x = it.get("from_x")
            if not from_x:
                from_x = (self.lookup_by_ed(from_ed) or {}).get("x25519_pub")
            text = None
            if from_x:
                try: text = unseal(self.id.x_priv, bytes.fromhex(from_x), bytes.fromhex(it["nonce"]), bytes.fromhex(it["ciphertext"])).decode("utf-8", "replace")
                except Exception: text = None
            item={"id": it.get("id"), "from_ed": from_ed, "from_x": from_x, "ts_ms": it.get("ts_ms"), "raw": text, "text": clean_wire_body(text) if text is not None else None}
            if verbose:print("[amni-chat] drained item from",from_ed,"len",len(text or "") if text else 0)
            out.append(item)
        return out
def run_relay(client, on_message, interval=3.0, should_stop=None, on_error=None, verbose=False, seen_path=None, workers=0, on_ack=None):
    import time,hashlib
    from collections import deque
    _seen=set();_seen_q=deque()
    if seen_path and os.path.exists(seen_path):
        try:
            _ids=[l.strip() for l in open(seen_path,encoding='utf-8') if l.strip()][-4000:]
            _seen=set(_ids);_seen_q=deque(_ids)
            open(seen_path,'w',encoding='utf-8').write('\n'.join(_ids)+'\n')
        except Exception:pass
    def _process(it):
        reply=on_message(it)
        if not reply:return
        text,images=(reply,[]) if isinstance(reply,str) else ((reply.get("text") or reply.get("reply") or ""),(reply.get("images") or []))
        if text:client.send_text(it["from_ed"],it["from_x"],text)
        for p in images:
            try:
                with open(p,"rb") as _f:data=_f.read()
                nm=os.path.basename(p);mime="image/png" if nm.lower().endswith(".png") else "application/octet-stream"
                client.send_attachment(it["from_ed"],it["from_x"],data,mime,nm)
            except Exception as _ie:(on_error or (lambda _e:None))(_ie)
    wq=None;pending=[0]
    if workers and workers>0:
        import threading,queue as _que
        from collections import defaultdict
        wq=_que.Queue();plocks=defaultdict(threading.Lock)
        def _worker():
            while True:
                it=wq.get()
                try:
                    with plocks[it["from_ed"]]:_process(it)
                except Exception as e:(on_error or (lambda _e:None))(e)
                finally:pending[0]-=1;wq.task_done()
        for _ in range(int(workers)):threading.Thread(target=_worker,daemon=True,name='amni-chat-worker').start()
    while not (should_stop and should_stop()):
        try:
            for it in client.drain(verbose=verbose):
                if (it.get("text") is None and not str(it.get("raw") or "").startswith("\x04")) or not it.get("from_x"): continue
                _mid=it.get("id") or hashlib.sha1((str(it.get("from_ed",""))+"|"+str(it.get("ts_ms",""))+"|"+str(it.get("text",""))).encode("utf-8","replace")).hexdigest()
                if _mid in _seen: continue
                _seen.add(_mid); _seen_q.append(_mid)
                if len(_seen_q)>4000: _seen.discard(_seen_q.popleft())
                if seen_path:
                    try:open(seen_path,'a',encoding='utf-8').write(_mid+'\n')
                    except Exception:pass
                if wq is None:
                    _process(it)
                else:
                    ahead=pending[0];pending[0]+=1;wq.put(it)
                    if ahead>0 and on_ack:
                        try:
                            _am=on_ack(it,ahead)
                            if _am:client.send_text(it["from_ed"],it["from_x"],_am)
                        except Exception:pass
        except Exception as e:
            (on_error or (lambda _e: None))(e)
        time.sleep(interval)
def _smoke():
    print("backend:", crypto_backend())
    a = Identity.generate(); b = Identity.generate()
    msg = "Rao! walkie check 📻 — GF(17) lives".encode("utf-8")
    nonce, ct = seal(a.x_priv, b.x_pub, msg)
    back = unseal(b.x_priv, a.x_pub, nonce, ct)
    assert back == msg, "round-trip mismatch"
    assert _x_shared(a.x_priv, b.x_pub) == _x_shared(b.x_priv, a.x_pub), "ECDH asymmetry"
    assert _hkdf_sha256(b"test", 32).hex() == hmac.new(hmac.new(_HKDF_SALT, b"test", hashlib.sha256).digest(), INFO + b"\x01", hashlib.sha256).digest().hex(), "hkdf"
    pl = parse_profile_link("Add me https://chat.amni-scient.com/add?p=%2B15551234567&ed=" + "ab" * 32 + "&x=" + "cd" * 32 + "&fp=x&np=" + "ef" * 32)
    assert pl and pl["ed"] == "ab" * 32 and pl["x"] == "cd" * 32, "profile link parse"
    assert clean_wire_body("\x01R:abc|quoted\nhello there") == "hello there", "reply unwrap"
    assert clean_wire_body("\x01R:abc|quoted") == "abc|quoted", "reply no-body"
    assert clean_wire_body("\x04blob:key:nonce:image/png:c.png") is None, "attachment frame"
    assert clean_wire_body("\tmsgid123") is None, "ack frame"
    assert clean_wire_body("\x02mid\x01\U0001F44D") is None, "reaction frame"
    assert clean_wire_body("\x081719944243123") is None, "read receipt"
    assert clean_wire_body("\x18mid\x1fnew body") is None, "edit frame"
    assert clean_wire_body("\x1aAnthony\x1chello") == "hello", "name unwrap"
    assert clean_wire_body("\x1aAzno\x1c\x04b:k:n:m:f") is None, "wrapped attachment"
    assert clean_wire_body("PLUG 15m") == "PLUG 15m", "plain body"
    print("OK envelope round-trip, ECDH symmetry, HKDF, profile-link, wire-clean all pass")
if __name__ == "__main__":
    _smoke()
