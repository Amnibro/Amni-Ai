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
const NS = 40u;
const MAXB = 256u;
const WM = 1048575u;
const BASE = 21u;
const TE = $TEu;
const QT = $QT;
const HB = $HBu;
const TC = $TC;
const TAU = 6.283185307179586;
const PI = 3.1415927410e+00;
const PILO = -8.7422776573e-08;
const WORD = array<u32, 8>($WORD);
const PUNC = array<u32, 8>($PUNC);
var<workgroup> pl: array<vec4<u32>, 32>;
var<workgroup> tx: array<vec2<u32>, 32>;
var<workgroup> np: u32;
var<workgroup> bq: vec2<u32>;
var<workgroup> lg: array<f32, 256>;
var<workgroup> tp: array<u32, 3>;
var<workgroup> hx: vec2<u32>;
fn hm(a: u32, b: u32, s: u32) -> u32 { var h = (a * 73856093u) ^ (b * 19349663u) ^ (s * 83492791u); h ^= h >> 13u; h *= 0x5BD1E995u; return h ^ (h >> 15u); }
fn sl(u: u32, v: u32, l: u32) -> u32 { return hm(u, v, l + 1u) % TE; }
fn cl(l: u32, e: u32) -> vec2<u32> { return textureLoad(cells, vec2<u32>(e % 4096u, e / 4096u), l, 0).xy; }
fn dp(r: u32, c: u32) -> f32 { return textureLoad(dpt, vec2<u32>((r % 16u) * 256u + c, r / 16u), 0).x; }
fn bt(m: array<u32, 8>, c: u32) -> bool { var q = m; return ((q[c >> 5u] >> (c & 31u)) & 1u) == 1u; }
fn ad(l: u32, u: u32, v: u32, w: f32) { pl[np] = vec4<u32>(l, sl(u, v, l), bitcast<u32>(w), 0u); np += 1u; }
fn sw(t: vec2<u32>, s: u32) -> f32 {
    let u = t.x >> 24u;
    let d = i32(select(select(u % 5u, (u / 5u) % 5u, s == 1u), u / 25u, s == 0u)) - 2;
    let lv = unpack2x16float(t.y);
    let m = select(select(0.0, lv.x, abs(d) == 1), lv.y, abs(d) == 2);
    return select(m, -m, d < 0);
}
fn atf(x0: f32) -> f32 {
    let ix = bitcast<u32>(x0) & 0x7fffffffu;
    let neg = (bitcast<u32>(x0) >> 31u) == 1u;
    if (ix >= 0x4c800000u) { return select(1.5707963705062866, -1.5707963705062866, neg); }
    if (ix < 0x39800000u) { return x0; }
    var x = x0;
    var id = -1;
    if (ix >= 0x3ee00000u) {
        x = abs(x0);
        if (ix < 0x3f300000u) { id = 0; x = (2.0 * x - 1.0) / (2.0 + x); } else if (ix < 0x3f980000u) { id = 1; x = (x - 1.0) / (x + 1.0); } else if (ix < 0x401c0000u) { id = 2; x = (x - 1.5) / (1.0 + 1.5 * x); } else { id = 3; x = -1.0 / x; }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * (3.3333328366e-01 + w * (1.4253635705e-01 + w * 6.1687607318e-02));
    let s2 = w * (-1.9999158382e-01 + w * -1.0648017377e-01);
    if (id < 0) { return x - x * (s1 + s2); }
    var hi = array<f32, 4>(4.6364760399e-01, 7.8539812565e-01, 9.8279368877e-01, 1.5707962513e+00);
    var lw = array<f32, 4>(5.0121582440e-09, 3.7748947079e-08, 3.4473217170e-08, 7.5497894159e-08);
    let r = hi[id] - ((x * (s1 + s2) - lw[id]) - x);
    return select(r, -r, neg);
}
fn at2(y: f32, x: f32) -> f32 {
    let ix0 = bitcast<u32>(x);
    let iy0 = bitcast<u32>(y);
    if (ix0 == 0x3f800000u) { return atf(y); }
    let m = ((iy0 >> 31u) & 1u) | ((ix0 >> 30u) & 2u);
    let ix = ix0 & 0x7fffffffu;
    let iy = iy0 & 0x7fffffffu;
    if (iy == 0u) { return select(y, select(-PI, PI, m == 2u), m >= 2u); }
    if (ix == 0u || ix + (26u << 23u) < iy) { return select(PI * 0.5, -PI * 0.5, (m & 1u) == 1u); }
    let z = select(atf(abs(y / x)), 0.0, (m & 2u) != 0u && iy + (26u << 23u) < ix);
    return select(select(select((z - PILO) - PI, PI - (z - PILO), m == 2u), -z, m == 1u), z, m == 0u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    let b = wg.x;
    if (c == 0u) {
        let q = io[p.io + b];
        let bb = q >> 16u;
        let sb = (p.src * MAXB + (q & 0xFFFFu)) * NS;
        let db = (p.dst * MAXB + b) * NS;
        var s: array<u32, 40>;
        for (var i = 0u; i < NS; i++) { s[i] = st[sb + i]; }
        var fx = bitcast<f32>(s[37]);
        var fy = bitcast<f32>(s[38]);
        var wh = s[16]; var pw1 = s[17]; var pw2 = s[18]; var pw3 = s[19]; var pw4 = s[20]; var s1 = s[21]; var s2 = s[22]; var g1 = s[23]; var at = s[24]; var ind = s[25]; var g2 = s[26]; var col = s[27]; var inc = s[28];
        var lastp = s[29] & 0xFFFFu; var tick = s[29] >> 16u;
        var inq = s[30]; var q1 = s[31]; var q2 = s[32]; var c1 = s[33]; var c2 = s[34]; var qon = s[35]; var wlen = s[36];
        let isw = bt(WORD, bb);
        if (!isw && wh != 0u) {
            pw4 = pw3; pw3 = pw2; pw2 = pw1; pw1 = wh; s2 = s1; g2 = g1;
            let lx = textureLoad(lex, vec2<u32>((pw1 & WM) % 1024u, (pw1 & WM) / 1024u), 0);
            let tr = textureLoad(trg, vec2<u32>(pw1 % 360u, 0u), 0);
            s1 = lx.x; g1 = lx.y; fx = 0.99 * fx + tr.x * TC * 8.0; fy = 0.99 * fy + tr.y * TC * 8.0;
            if (QT && inq == 1u && wlen >= 3u && !(bb == 58u && s[0] == 109u && s[1] == 97u && s[2] == 100u && s[3] == 65u) && lx.z > 0u) {
                if (c1 == 0u || lx.z < c1) { q2 = q1; c2 = c1; q1 = pw1; c1 = lx.z; } else if (pw1 != q1 && (c2 == 0u || lx.z < c2)) { q2 = pw1; c2 = lx.z; }
            }
        }
        wlen = select(wlen, select(0u, wlen + 1u, isw), QT);
        wh = select(0u, wh * 83492791u + bb + 1u, isw);
        let nl = bb == 10u;
        col = select(col + 1u, 0u, nl);
        if (nl) { at = 1u; ind = 0u; } else if (bb == 32u) { ind += select(0u, 1u, at == 1u); } else { at = 0u; }
        tick = select(select(tick, 0u, nl), tick + 1u, bb == 96u);
        if (tick == 3u) { inc = 1u - inc; tick = 0u; }
        lastp = select(select(lastp, 10u, nl), bb, bt(PUNC, bb));
        for (var i = 15u; i > 0u; i--) { s[i] = s[i - 1u]; }
        s[0] = bb;
        if (QT) {
            if (s[5] == 85u && s[4] == 115u && s[3] == 101u && s[2] == 114u && s[1] == 58u && s[0] == 32u) { q1 = 0u; q2 = 0u; c1 = 0u; c2 = 0u; inq = 1u; qon = 0u; }
            if (s[5] == 65u && s[4] == 100u && s[3] == 97u && s[2] == 109u && s[1] == 58u && s[0] == 32u) { inq = 0u; qon = select(0u, 1u, q1 != 0u); }
            if (s[0] == 10u && s[1] == 10u) { qon = 0u; }
        }
        s[16] = wh; s[17] = pw1; s[18] = pw2; s[19] = pw3; s[20] = pw4; s[21] = s1; s[22] = s2; s[23] = g1; s[24] = at; s[25] = ind; s[26] = g2; s[27] = col; s[28] = inc; s[29] = lastp | (tick << 16u);
        s[30] = inq; s[31] = q1; s[32] = q2; s[33] = c1; s[34] = c2; s[35] = qon; s[36] = wlen; s[37] = bitcast<u32>(fx); s[38] = bitcast<u32>(fy);
        for (var i = 0u; i < NS; i++) { st[db + i] = s[i]; }
        if (p.want == 1u) {
            np = 0u;
            bq = vec2<u32>(s[0], s[1]);
            ad(2u, hm(s[2], s[1], 7u), s[0], 1.0);
            ad(3u, hm(hm(s[3], s[2], 9u), s[1], 11u), s[0], 1.0);
            ad(4u, wh, pw1 & 255u, 1.0);
            ad(5u, pw1 & 4095u, pw2 & 4095u, 1.0);
            ad(6u, hm(min(ind, 32u), inc * 512u + lastp, 13u), min(col, 255u), 1.0);
            let h5 = hm(hm(s[4], s[3], 17u), hm(s[2], s[1], 19u), 21u);
            let h6 = hm(h5, s[5], 23u);
            let h8 = hm(hm(h6, s[6], 29u), s[7], 31u);
            let h10 = hm(hm(h8, s[8], 41u), s[9], 43u);
            let h12 = hm(hm(h10, s[10], 47u), s[11], 53u);
            let h16 = hm(hm(hm(hm(h12, s[12], 59u), s[13], 61u), s[14], 67u), s[15], 71u);
            ad(7u, h5, s[0], 1.0); ad(8u, h6, s[0], 1.0); ad(9u, h8, s[0], 1.0); ad(10u, hm(pw1, pw2, 37u) & 65535u, pw3 & 4095u, 1.0); ad(11u, h10, s[0], 1.0); ad(12u, h12, s[0], 1.0); ad(13u, h16, s[0], 1.0);
            ad(14u, hm(hm(pw1, pw2, 73u), hm(pw3, pw4, 79u), 83u), wh & 65535u, 1.0); ad(15u, wh, 0u, 1.0); ad(16u, hm(wh, pw1, 89u), 0u, 1.0); ad(17u, s1, s2, 1.0); ad(18u, g1, g2, 1.0); ad(19u, wh, s1, 1.0);
            let a0 = at2(fy, fx) / TAU;
            let uu = (a0 - floor(a0)) * 256.0;
            let vv = min(sqrt(fx * fx + fy * fy), 7.99) / 8.0 * 256.0;
            let u0 = floor(uu);
            let v0 = floor(vv);
            let fu = uu - u0;
            let fv = vv - v0;
            for (var k = 0; k < 4; k++) {
                let w = select(1.0 - fu, fu, (k & 1) == 1) * select(1.0 - fv, fv, (k >> 1u) == 1);
                if (w != 0.0) { ad(BASE - 1u, u32((i32(u0) + (k & 1)) % 256), u32(min(i32(v0) + (k >> 1u), 255)), w); }
            }
            if (QT && qon == 1u) { ad(BASE, q1, wh, 1.0); ad(BASE + 1u, hm(q1, q2, 107u), pw1 & 4095u, 1.0); ad(BASE + 2u, q1, s[0] + (s[1] << 8u), 1.0); }
        }
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
    for (var h = 0u; h < 4u; h++) {
        lg[c] = a;
        workgroupBarrier();
        var r = 0u;
        for (var d = 0u; d < 256u; d++) { let u = lg[d]; r += select(0u, 1u, u > a || (u == a && d < c)); }
        if (r < 3u) { tp[r] = c; }
        workgroupBarrier();
        if (c == 0u) {
            let x = tp[0]; let y = tp[1]; let z = tp[2];
            let l3 = min(x, min(y, z)); let h3 = max(x, max(y, z));
            hx = cl(HB + h, sl(hm(hm(l3 * 65536u + (x + y + z - l3 - h3) * 256u + h3, b1, 97u + h), b2, 103u + h), h, BASE + h));
        }
        workgroupBarrier();
        let t = hx;
        for (var s = 0u; s < 3u; s++) { if (((t.x >> (8u * s)) & 255u) == c) { a += sw(t, s); } }
    }
    lo[p.out + b * 256u + c] = a;
}
