struct P { n: u32, t: u32, salt: u32, mode: u32, it: u32, lr: f32, c1: f32, c2: f32, o1: f32, o2: f32, x: u32, y: u32 }
var<push_constant> p: P;
const LV = $LVu;
const HOPS = $HOPSu;
const NT = (C0 - 2u) * 4u;
const COPY = $COPYu;
const CG = 4.0;
const TERN = $TERNu;
const QLEV = $QLEVu;
const TT = $TT;
const SG = $SG;
const SLO = $SLO;
const SHI = $SHI;
const RTH = $RTH;
const RW = $RW;
const WD = $WD;
const DE = 65536u;
fn tx(e: u32) -> vec2<u32> { return vec2<u32>(e % 4096u, e / 4096u); }
fn dx(r: u32, c: u32) -> vec2<u32> { return vec2<u32>((r % 16u) * 256u + c, r / 16u); }
fn gi(wg: vec3<u32>, li: u32) -> u32 { return (wg.y * 32768u + wg.x) * 256u + li; }
fn nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn fx(v: f32) -> vec2<u32> {
    let b = bitcast<u32>(v);
    let ex = i32((b >> 23u) & 255u);
    let m = (b & 0x7fffffu) | 0x800000u;
    let sh = ex - 100;
    if (ex == 0 || sh <= -24) { return vec2<u32>(0u); }
    let x = select(select(vec2<u32>(0u, m << u32(max(sh - 32, 0))), vec2<u32>(m << u32(max(sh, 0)), select(0u, m >> u32(clamp(32 - sh, 1, 31)), sh > 0)), sh < 32), vec2<u32>(m >> u32(max(-sh, 0)), 0u), sh < 0);
    return select(x, vec2<u32>(~x.x + 1u, ~x.y + select(0u, 1u, x.x == 0u)), (b >> 31u) == 1u);
}
fn fxf(a: vec2<u32>) -> f32 {
    let neg = (a.y >> 31u) == 1u;
    let r = (f32(select(a.y, ~a.y + select(0u, 1u, a.x == 0u), neg)) * 4294967296.0 + f32(select(a.x, ~a.x + 1u, neg))) * 8.881784197001252e-16;
    return select(r, -r, neg);
}
