@group(0) @binding(0) var swt: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> lq: array<f32>;
var<workgroup> ra: array<f32, 256>;
var<workgroup> rn: array<u32, 256>;
fn red(c: u32, a: f32, n: u32) -> vec2<f32> {
    ra[c] = a;
    rn[c] = n;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (c < s) { ra[c] += ra[c + s]; rn[c] += rn[c + s]; } workgroupBarrier(); }
    let r = vec2<f32>(ra[0], f32(rn[0]));
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    let l = wg.x;
    var a = 0.0;
    for (var i = c; i < TE * 3u; i += 256u) { a += abs(textureLoad(swt, tx(i / 3u), l * 3u + i % 3u, 0).x); }
    let th = TT * (red(c, a, 0u).x / f32(TE * 3u));
    var acc = 0.0;
    var n = 0u;
    for (var i = c; i < TE * 3u; i += 256u) { let x = abs(textureLoad(swt, tx(i / 3u), l * 3u + i % 3u, 0).x); if (x >= th) { acc += x; n += 1u; } }
    let r = red(c, acc, n);
    if (c == 0u) { lq[l] = th; lq[LV + l] = r.x / max(r.y, 1.0); }
}
