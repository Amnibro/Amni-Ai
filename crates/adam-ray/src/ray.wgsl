struct P { nb: u32, src: u32, dst: u32, want: u32, io: u32, out: u32 }
var<push_constant> p: P;
@group(0) @binding(0) var cells: texture_2d_array<u32>;
@group(0) @binding(1) var dpt: texture_2d<f32>;
@group(0) @binding(2) var lex: texture_2d<u32>;
@group(0) @binding(3) var bit: texture_2d<f32>;
@group(0) @binding(4) var trg: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> st: array<u32>;
@group(0) @binding(6) var<storage, read> io: array<u32>;
@group(0) @binding(7) var<storage, read_write> lo: array<f32>;
@group(0) @binding(8) var<storage, read> ex: array<u32>;
const V19 = $V19;
const EX = $EXu;
const MAXB = 256u;
var<workgroup> pl: array<vec4<u32>, 32>;
var<workgroup> tx: array<vec2<u32>, 32>;
var<workgroup> np: u32;
var<workgroup> bq: vec2<u32>;
var<workgroup> lg: array<f32, 256>;
var<workgroup> tp: array<u32, 3>;
var<workgroup> hx: vec2<u32>;
fn cl(l: u32, e: u32) -> vec2<u32> { return textureLoad(cells, vec2<u32>(e % 4096u, e / 4096u), l, 0).xy; }
fn dp(r: u32, c: u32) -> f32 { return textureLoad(dpt, vec2<u32>((r % 16u) * 256u + c, r / 16u), 0).x; }
fn put(l: u32, k: u32, e: u32, w: f32) { pl[np] = vec4<u32>(l, e, bitcast<u32>(w), 0u); np += 1u; }
fn sw(t: vec2<u32>, s: u32) -> f32 {
    let u = t.x >> 24u;
    let d = i32(select(select(u % 5u, (u / 5u) % 5u, s == 1u), u / 25u, s == 0u)) - 2;
    let lv = unpack2x16float(t.y);
    let m = select(select(0.0, lv.x, abs(d) == 1), lv.y, abs(d) == 2);
    return select(m, -m, d < 0);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    let b = wg.x;
    if (c == 0u) {
        var s: array<u32, 40>;
        if (V19) { for (var i = 0u; i < NS; i++) { s[i] = ex[(p.io + b) * EX + i]; } } else {
            let q = io[p.io + b];
            let sb = (p.src * MAXB + (q & 0xFFFFu)) * NS;
            let db = (p.dst * MAXB + b) * NS;
            for (var i = 0u; i < NS; i++) { s[i] = st[sb + i]; }
            upd(&s, q >> 16u);
            for (var i = 0u; i < NS; i++) { st[db + i] = s[i]; }
        }
        if (p.want == 1u) { np = 0u; bq = vec2<u32>(s[0], s[1]); plan(&s); }
    }
    workgroupBarrier();
    if (p.want == 0u) { return; }
    let n = workgroupUniformLoad(&np);
    if (c < n) { tx[c] = cl(pl[c].x, pl[c].y); }
    workgroupBarrier();
    let b1 = bq.x;
    let b2 = bq.y;
    var a = textureLoad(bit, vec2<u32>(c, 0u), 0).x + dp(b1, c) + dp(256u + b2 * 256u + b1, c);
    for (var k = 0u; k < n; k++) {
        let t = tx[k];
        let w = bitcast<f32>(pl[k].z);
        for (var s = 0u; s < 3u; s++) { if (((t.x >> (8u * s)) & 255u) == c) { a += w * sw(t, s); } }
    }
    if (V19) { let o = (p.io + b) * EX + NS; for (var k = 0u; k < 13u; k++) { if (ex[o + 2u * k] == c) { a += bitcast<f32>(ex[o + 2u * k + 1u]); } } }
    for (var h = 0u; h < 4u; h++) {
        lg[c] = a;
        workgroupBarrier();
        var r = 0u;
        for (var d = 0u; d < 256u; d++) { let u = lg[d]; r += select(0u, 1u, u > a || (u == a && d < c)); }
        if (r < 3u) { tp[r] = c; }
        workgroupBarrier();
        if (c == 0u) {
            hx = cl(HB + h, hkey(vec3<u32>(tp[0], tp[1], tp[2]), b1, b2, h));
        }
        workgroupBarrier();
        let t = hx;
        for (var s = 0u; s < 3u; s++) { if (((t.x >> (8u * s)) & 255u) == c) { a += sw(t, s); } }
    }
    lo[p.out + b * 256u + c] = a;
}
