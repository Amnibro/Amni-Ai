const NS = 40u;
const WM = 1048575u;
const TE = $TEu;
const QT = $QT;
const BASE = $BASEu;
const LVH = $LVHu;
const HB = $HBu;
const C0 = $C0u;
const TA = $TA;
const TC = $TC;
const TAU = 6.283185307179586;
const PI = 3.1415927410e+00;
const PILO = -8.7422776573e-08;
const WORD = array<u32, 8>($WORD);
const PUNC = array<u32, 8>($PUNC);
var<private> salt: u32;
fn hm(a: u32, b: u32, s: u32) -> u32 { var h = (a * 73856093u) ^ (b * 19349663u) ^ (s * 83492791u); h ^= h >> 13u; h *= 0x5BD1E995u; return h ^ (h >> 15u); }
fn sl(u: u32, v: u32, l: u32) -> u32 { return hm(u, v, l + 1u + salt * 1000003u) % TE; }
fn bt(m: array<u32, 8>, c: u32) -> bool { var q = m; return ((q[c >> 5u] >> (c & 31u)) & 1u) == 1u; }
fn has(l: u32) -> bool { return l <= 6u || (l <= 10u && LVH >= 12u) || (l <= 14u && LVH >= 16u) || (l <= 16u && BASE >= 18u) || (l <= 19u && BASE >= 21u); }
fn hkey(t: vec3<u32>, b1: u32, b2: u32, h: u32) -> u32 { let l3 = min(t.x, min(t.y, t.z)); let h3 = max(t.x, max(t.y, t.z)); return sl(hm(hm(l3 * 65536u + (t.x + t.y + t.z - l3 - h3) * 256u + h3, b1, 97u + h), b2, 103u + h), h, BASE + h); }
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
