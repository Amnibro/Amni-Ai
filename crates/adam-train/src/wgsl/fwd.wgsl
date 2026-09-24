@group(0) @binding(0) var dpt: texture_2d<f32>;
@group(0) @binding(1) var sidt: texture_2d_array<u32>;
@group(0) @binding(2) var swt: texture_2d_array<f32>;
@group(0) @binding(3) var bit: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> lq: array<f32>;
@group(0) @binding(5) var<storage, read> xy: array<u32>;
@group(0) @binding(6) var<storage, read_write> pi: array<u32>;
@group(0) @binding(7) var<storage, read_write> pw: array<f32>;
@group(0) @binding(8) var<storage, read_write> dl: array<f32>;
@group(0) @binding(9) var<storage, read_write> pl: array<f32>;
@group(0) @binding(10) var<storage, read> cpb: array<i32>;
var<workgroup> tw: array<f32, NT>;
var<workgroup> tid: array<u32, NT>;
var<workgroup> tq: array<vec3<f32>, NT>;
var<workgroup> lg: array<f32, 256>;
var<workgroup> ex: array<f32, 256>;
var<workgroup> tv: array<vec3<f32>, 256>;
var<workgroup> tj: array<vec3<u32>, 256>;
var<workgroup> hid: u32;
var<workgroup> hq: vec3<f32>;
var<workgroup> ps: array<f32, 8>;
var<workgroup> zs: f32;
var<workgroup> cgv: array<f32, 16>;
var<workgroup> cbv: array<i32, 16>;
fn qv(l: u32, w: vec3<f32>) -> vec3<f32> {
    if (TERN == 0u) { return w; }
    let a = abs(w);
    if (TERN == 2u) { return select(select(-vec3<f32>(lq[LV + l]), vec3<f32>(lq[LV + l]), w > vec3<f32>(0.0)), vec3<f32>(0.0), a < vec3<f32>(lq[l])); }
    let mx = max(max(max(0.0, a.x), a.y), a.z);
    let lg2 = log(max(mx, 1e-30)) * 1.4426950408889634;
    let mq = select(mx, select(pow(2.0, SLO + round((min(lg2, SHI) - SLO) / (SHI - SLO) * 14.0) * (SHI - SLO) / 14.0), 0.0, lg2 < SLO), SG);
    let m5 = select(select(vec3<f32>(mq), vec3<f32>(0.5 * mq), a < vec3<f32>(0.75 * mx)), vec3<f32>(mq), QLEV == 3u);
    return select(select(-m5, m5, w > vec3<f32>(0.0)), vec3<f32>(0.0), a < vec3<f32>(TT * mx));
}
fn cq(l: u32, e: u32) -> vec3<f32> { let c = tx(e); return qv(l, vec3<f32>(textureLoad(swt, c, l * 3u, 0).x, textureLoad(swt, c, l * 3u + 1u, 0).x, textureLoad(swt, c, l * 3u + 2u, 0).x)); }
fn hit(id: u32, v: vec3<f32>, c: u32, a0: f32, w: f32) -> f32 {
    var a = a0;
    for (var s = 0u; s < 3u; s++) { if (((id >> (8u * s)) & 255u) == c) { a += w * v[s]; } }
    return a;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    let q = wg.y * 32768u + wg.x;
    if (q >= p.n) { return; }
    salt = p.salt;
    let base = q * LV * 4u;
    for (var i = c; i < NT; i += 256u) {
        let j = base + 8u + i;
        let w = pw[j];
        tw[i] = w;
        if (w != 0.0) { let l = 2u + i / 4u; let e = pi[j]; tid[i] = textureLoad(sidt, tx(e), l, 0).x; tq[i] = cq(l, e); }
    }
    workgroupBarrier();
    let b1 = pi[base];
    let e1 = pi[base + 4u];
    var a = textureLoad(bit, vec2<u32>(c, 0u), 0).x + pw[base] * textureLoad(dpt, dx(b1, c), 0).x + pw[base + 4u] * textureLoad(dpt, dx(DE + e1, c), 0).x;
    for (var k = 0u; k < NT; k++) { let w = tw[k]; if (w != 0.0) { a = hit(tid[k], tq[k], c, a, w); } }
    if (COPY > 0u) {
        if (c < COPY) {
            let cb = cpb[q * COPY + c];
            let l = C0 + c;
            let j = base + l * 4u;
            cbv[c] = cb;
            if (cb >= 0) {
                var g = 0.0;
                for (var i = 0u; i < select(1u, 4u, COPY >= 13u); i++) { g += pw[j + i] * textureLoad(swt, tx(pi[j + i]), l * 3u, 0).x; }
                cgv[c] = select(CG * pw[j] * textureLoad(swt, tx(pi[j]), l * 3u, 0).x, CG * g, COPY >= 13u);
            }
        }
        workgroupBarrier();
        for (var k = 0u; k < COPY; k++) { if (cbv[k] == i32(c)) { a += cgv[k]; } }
    }
    for (var h = 0u; h < HOPS; h++) {
        tv[c] = vec3<f32>(a, -3.4e38, -3.4e38);
        tj[c] = vec3<u32>(c, 0xffffffffu, 0xffffffffu);
        workgroupBarrier();
        for (var s = 128u; s > 0u; s >>= 1u) {
            if (c < s) {
                let av = tv[c]; let ai = tj[c]; let bv = tv[c + s]; let bi = tj[c + s];
                var rv: vec3<f32>; var ri: vec3<u32>; var x = 0u; var y = 0u;
                for (var k = 0u; k < 3u; k++) { let ta = av[x] > bv[y] || (av[x] == bv[y] && ai[x] < bi[y]); rv[k] = select(bv[y], av[x], ta); ri[k] = select(bi[y], ai[x], ta); x += select(0u, 1u, ta); y += select(1u, 0u, ta); }
                tv[c] = rv; tj[c] = ri;
            }
            workgroupBarrier();
        }
        if (c == 0u) {
            let l = HB + h;
            let e = hkey(tj[0], b1, e1 / 256u, h);
            pi[base + l * 4u] = e;
            pw[base + l * 4u] = 1.0;
            hid = textureLoad(sidt, tx(e), l, 0).x;
            hq = cq(l, e);
        }
        workgroupBarrier();
        a = hit(hid, hq, c, a, 1.0);
    }
    if (p.mode == 2u) { dl[q * 256u + c] = a; return; }
    lg[c] = a;
    ex[c] = a;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (c < s) { ex[c] = max(ex[c], ex[c + s]); } workgroupBarrier(); }
    let mx = max(ex[0], -1e30);
    workgroupBarrier();
    ex[c] = exp(a - mx);
    workgroupBarrier();
    if (c < 8u) { var z = 0.0; for (var i = 0u; i < 32u; i++) { z += ex[c * 32u + i]; } ps[c] = z; }
    workgroupBarrier();
    let y = (xy[q] >> 8u) & 255u;
    if (c == 0u) { var z = 0.0; for (var g = 0u; g < 8u; g++) { z += ps[g]; } zs = z; pl[q] = -((lg[y] - mx) - log(z)); }
    workgroupBarrier();
    if (p.mode == 0u) { dl[q * 256u + c] = (ex[c] / zs - select(0.0, 1.0, c == y)) / f32(p.n); }
}
