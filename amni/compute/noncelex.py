import numpy as np, hashlib, hmac as _hmac, struct, re, time
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from collections import Counter
P = 17
MODE_NONCELEX = 30
try:
    from amni.compute.prismtex import P as _P, MODE_NONCELEX as _M
    P, MODE_NONCELEX = _P, _M
except Exception:
    pass
P2, P3 = P * P, P * P * P
MAX_NONCE = P * P * P * P - 1
CHAR_MAX = 127
WORD_BASE = 128
NLX_MAGIC = 0x4E4C5850
NLX_VER = 1
_WORD_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[ \t]+|\n|.)')
def nonce_to_rgba(nid: int) -> np.ndarray:
    v = np.uint32(nid)
    return np.array([v % P, (v // P) % P, (v // P2) % P, (v // P3) % P], dtype=np.uint8)
def rgba_to_nonce(px: np.ndarray) -> int:
    return int(px[0]) + int(px[1]) * P + int(px[2]) * P2 + int(px[3]) * P3
def nonce_to_rgba_batch(nids: np.ndarray) -> np.ndarray:
    v = nids.astype(np.uint32)
    return np.stack([v % P, (v // P) % P, (v // P2) % P, (v // P3) % P], axis=-1).astype(np.uint8)
def rgba_to_nonce_batch(px: np.ndarray) -> np.ndarray:
    p = px.astype(np.uint32)
    return p[..., 0] + p[..., 1] * P + p[..., 2] * P2 + p[..., 3] * P3
class NonceLexCodec:
    __slots__ = ('_w2id', '_id2w', '_next_id')
    def __init__(self):
        self._w2id: Dict[str, int] = {}
        self._id2w: List[str] = [chr(i) if i > 0 else '' for i in range(WORD_BASE)]
        self._next_id = WORD_BASE
    def _char_nonce(self, ch: str) -> int:
        o = ord(ch)
        return o if 0 < o <= CHAR_MAX else 0
    def _add_word(self, w: str) -> int:
        if w in self._w2id:
            return self._w2id[w]
        nid = self._next_id
        if nid > MAX_NONCE:
            return -1
        self._w2id[w] = nid
        self._id2w.append(w) if nid == len(self._id2w) else None
        self._next_id = nid + 1
        return nid
    def build_vocab(self, text: str) -> int:
        tokens = _WORD_RE.findall(text)
        freq = Counter()
        for t in tokens:
            freq[t] += (1 if len(t) > 1 or not t.isspace() else 0) if (len(t) > 1 and (t[0].isalpha() or t[0] == '_' or t[0].isdigit())) else 0
        for w, _ in freq.most_common():
            if w and len(w) > 1 and (w[0].isalpha() or w[0] == '_' or w[0].isdigit()):
                self._add_word(w)
        return self._next_id - WORD_BASE
    def build_vocab_multi(self, texts: List[str]) -> int:
        combined = '\n'.join(texts)
        return self.build_vocab(combined)
    @property
    def vocab_size(self) -> int:
        return self._next_id - WORD_BASE
    @property
    def total_nonces(self) -> int:
        return self._next_id
    def encode(self, text: str) -> np.ndarray:
        tokens = _WORD_RE.findall(text)
        nids = []
        for t in tokens:
            if t in self._w2id:
                nids.append(self._w2id[t])
            elif len(t) == 1:
                cn = self._char_nonce(t)
                if cn > 0:
                    nids.append(cn)
                else:
                    wid = self._add_word(t)
                    nids.append(wid if wid > 0 else 0)
            elif len(t) > 1 and t[0].isdigit():
                wid = self._add_word(t)
                if wid > 0:
                    nids.append(wid)
                else:
                    for ch in t:
                        cn = self._char_nonce(ch)
                        nids.append(cn if cn > 0 else 0)
            elif t.isspace():
                for ch in t:
                    nids.append(self._char_nonce(ch))
            else:
                for ch in t:
                    cn = self._char_nonce(ch)
                    if cn > 0:
                        nids.append(cn)
                    else:
                        wid = self._add_word(ch)
                        nids.append(wid if wid > 0 else 0)
        arr = np.array(nids, dtype=np.uint32) if nids else np.zeros(0, dtype=np.uint32)
        return nonce_to_rgba_batch(arr)
    def decode(self, pixels: np.ndarray) -> str:
        if pixels.size == 0:
            return ''
        nids = rgba_to_nonce_batch(pixels)
        parts = []
        for nid in nids:
            nid = int(nid)
            if nid == 0:
                continue
            elif nid <= CHAR_MAX:
                parts.append(chr(nid))
            elif nid < len(self._id2w):
                parts.append(self._id2w[nid])
            else:
                parts.append(f'<UNK:{nid}>')
        return ''.join(parts)
    def encode_lines(self, text: str, tex_width: int = 4096) -> np.ndarray:
        lines = text.split('\n')
        rows = []
        for line in lines:
            px = self.encode(line)
            n = px.shape[0]
            if n == 0:
                row = np.zeros((tex_width, 4), dtype=np.uint8)
            elif n <= tex_width:
                row = np.zeros((tex_width, 4), dtype=np.uint8)
                row[:n] = px
            else:
                n_rows = (n + tex_width - 1) // tex_width
                padded = np.zeros((n_rows * tex_width, 4), dtype=np.uint8)
                padded[:n] = px
                for r in range(n_rows):
                    rows.append(padded[r * tex_width:(r + 1) * tex_width])
                continue
            rows.append(row)
        return np.array(rows, dtype=np.uint8) if rows else np.zeros((1, tex_width, 4), dtype=np.uint8)
    def decode_lines(self, texture: np.ndarray) -> str:
        lines = []
        for row in texture:
            mask = row.any(axis=-1)
            last_nonzero = np.where(mask)[0]
            end = int(last_nonzero[-1]) + 1 if last_nonzero.size > 0 else 0
            lines.append(self.decode(row[:end]) if end > 0 else '')
        return '\n'.join(lines)
    def get_word_nonce(self, word: str) -> Optional[int]:
        return self._w2id.get(word)
    def get_nonce_word(self, nonce_id: int) -> Optional[str]:
        return self._id2w[nonce_id] if 0 < nonce_id < len(self._id2w) else None
    def _pack_vocab(self) -> bytes:
        parts = []
        for nid in range(WORD_BASE, self._next_id):
            w = self._id2w[nid]
            wb = w.encode('utf-8')
            parts.append(struct.pack('<H', len(wb)))
            parts.append(wb)
        return b''.join(parts)
    def _unpack_vocab(self, data: bytes, n_vocab: int):
        off = 0
        for _ in range(n_vocab):
            slen = struct.unpack('<H', data[off:off + 2])[0]
            off += 2
            w = data[off:off + slen].decode('utf-8')
            off += slen
            self._add_word(w)
    def compression_stats(self, text: str) -> Dict:
        px = self.encode(text)
        orig_bytes = len(text.encode('utf-8'))
        enc_bytes = px.shape[0] * 4
        vocab_bytes = len(self._pack_vocab())
        return {'original_bytes': orig_bytes, 'encoded_pixels': px.shape[0], 'encoded_bytes': enc_bytes, 'vocab_bytes': vocab_bytes, 'total_bytes': enc_bytes + vocab_bytes, 'ratio': (enc_bytes + vocab_bytes) / max(orig_bytes, 1), 'vocab_size': self.vocab_size, 'nonces_used': px.shape[0]}
NLX_VER_SIGNED = 2
def _get_signing_key(base: Path = None) -> bytes:
    base = base or Path(__file__).resolve().parent.parent.parent
    kf = base / ".amni_signing_key"
    if kf.exists(): return kf.read_bytes()[:32]
    key = hashlib.sha256(hashlib.sha256(struct.pack('<d', time.time()) + str(base).encode()).digest()).digest()
    kf.write_bytes(key)
    return key
def save_noncelex_ptex(path: str, codec: NonceLexCodec, content_pixels: np.ndarray, tex_width: int = 4096, signing_key: bytes = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    vocab_data = codec._pack_vocab()
    n_px = content_pixels.shape[0] if content_pixels.ndim == 2 else content_pixels.reshape(-1, 4).shape[0]
    flat_px = content_pixels.reshape(-1, 4)
    n_rows = (n_px + tex_width - 1) // tex_width
    page = np.zeros((n_rows, tex_width, 4), dtype=np.uint8)
    page.reshape(-1, 4)[:n_px] = flat_px[:n_px]
    payload = np.ascontiguousarray(page).tobytes()
    chk = hashlib.sha256(vocab_data + payload).digest()
    sk = signing_key or _get_signing_key(p.parent)
    sig = _hmac.new(sk, chk + vocab_data + payload, hashlib.sha256).digest()
    with open(str(p), 'wb') as f:
        f.write(struct.pack('<I', NLX_MAGIC))
        f.write(struct.pack('<HH', NLX_VER_SIGNED, MODE_NONCELEX))
        f.write(struct.pack('<I', codec.vocab_size))
        f.write(struct.pack('<I', n_px))
        f.write(struct.pack('<I', tex_width))
        f.write(struct.pack('<I', n_rows))
        f.write(chk)
        f.write(sig)
        f.write(struct.pack('<I', len(vocab_data)))
        f.write(vocab_data)
        f.write(payload)
def load_noncelex_ptex(path: str, signing_key: bytes = None, require_sig: bool = False) -> Tuple[NonceLexCodec, np.ndarray, Dict]:
    with open(path, 'rb') as f:
        mg = struct.unpack('<I', f.read(4))[0]
        assert mg == NLX_MAGIC, f"bad NonceLex magic: {mg:#x}"
        ver, mode = struct.unpack('<HH', f.read(4))
        n_vocab = struct.unpack('<I', f.read(4))[0]
        n_px = struct.unpack('<I', f.read(4))[0]
        tex_width = struct.unpack('<I', f.read(4))[0]
        n_rows = struct.unpack('<I', f.read(4))[0]
        stored_chk = f.read(32)
        stored_sig = f.read(32) if ver >= NLX_VER_SIGNED else None
        if ver < NLX_VER_SIGNED and require_sig:
            raise ValueError(f"PTEX v{ver} unsigned — signature required")
        vocab_len = struct.unpack('<I', f.read(4))[0]
        vocab_data = f.read(vocab_len)
        payload = f.read()
    actual_chk = hashlib.sha256(vocab_data + payload).digest()
    assert _hmac.compare_digest(actual_chk, stored_chk), "SHA-256 mismatch: file corrupted"
    if stored_sig is not None:
        sk = signing_key or _get_signing_key(Path(path).parent)
        expected_sig = _hmac.new(sk, stored_chk + vocab_data + payload, hashlib.sha256).digest()
        assert _hmac.compare_digest(stored_sig, expected_sig), "HMAC signature invalid: file tampered"
    codec = NonceLexCodec()
    codec._unpack_vocab(vocab_data, n_vocab)
    page = np.frombuffer(payload, dtype=np.uint8).reshape(n_rows, tex_width, 4)
    pixels = page.reshape(-1, 4)[:n_px]
    meta = {'version': ver, 'mode': mode, 'n_vocab': n_vocab, 'n_pixels': n_px, 'tex_width': tex_width, 'n_rows': n_rows, 'signed': stored_sig is not None}
    return codec, pixels, meta
def encode_file(filepath: str, codec: Optional[NonceLexCodec] = None) -> Tuple[NonceLexCodec, np.ndarray]:
    text = Path(filepath).read_text(encoding='utf-8', errors='replace')
    codec = codec or NonceLexCodec()
    codec.build_vocab(text)
    return codec, codec.encode(text)
def encode_context(texts: List[str], codec: Optional[NonceLexCodec] = None) -> Tuple[NonceLexCodec, List[np.ndarray]]:
    codec = codec or NonceLexCodec()
    codec.build_vocab_multi(texts)
    return codec, [codec.encode(t) for t in texts]
def encode_and_save(filepath: str, output_path: str, codec: Optional[NonceLexCodec] = None) -> Dict:
    codec, pixels = encode_file(filepath, codec)
    save_noncelex_ptex(output_path, codec, pixels)
    text = Path(filepath).read_text(encoding='utf-8', errors='replace')
    return codec.compression_stats(text)
def batch_encode_save(filepaths: List[str], output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    codec = NonceLexCodec()
    texts = {}
    for fp in filepaths:
        texts[fp] = Path(fp).read_text(encoding='utf-8', errors='replace')
    codec.build_vocab_multi(list(texts.values()))
    results = {}
    for fp, text in texts.items():
        px = codec.encode(text)
        name = Path(fp).stem + '.nlx.ptex'
        save_noncelex_ptex(str(out / name), codec, px)
        results[fp] = codec.compression_stats(text)
    return {'files': results, 'shared_vocab_size': codec.vocab_size, 'total_nonces': codec.total_nonces}
