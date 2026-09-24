@group(0) @binding(0) var<storage, read> ctr: array<u32>;
@group(0) @binding(1) var<storage, read> uk: array<u32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(3) var<storage, read> pi: array<u32>;
@group(0) @binding(4) var<storage, read> pw: array<f32>;
@group(0) @binding(5) var<storage, read> dl: array<f32>;
@group(0) @binding(6) var<storage, read> xy: array<u32>;
@group(0) @binding(7) var sidt: texture_2d_array<u32>;
@group(0) @binding(8) var<storage, read_write> fa: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> best: array<atomic<u32>>;
fn fadd(i: u32, v: f32) {
    let x = fx(v);
    if (x.x == 0u && x.y == 0u) { return; }
    let o = atomicAdd(&fa[i], x.x);
    let hi = x.y + select(0u, 1u, o + x.x < o);
    if (hi != 0u) { atomicAdd(&fa[i + 1u], hi); }
}
@compute @workgroup_size(256)
fn umap(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let u = gi(wg, li);
    if (u < ctr[0]) { cnt[uk[u]] = u; }
}
@compute @workgroup_size(256)
fn slots(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = gi(wg, li);
    if (j >= p.n * LV * 4u || pw[j] == 0.0) { return; }
    let l = (j / 4u) % LV;
    if (l < 2u) { return; }
    let e = pi[j];
    let u = cnt[l * TE + e];
    let q = j / (LV * 4u);
    let id = textureLoad(sidt, tx(e), l, 0).x;
    let y = (xy[q] >> 8u) & 255u;
    var miss = true;
    for (var s = 0u; s < 3u; s++) { let c = (id >> (8u * s)) & 255u; fadd((u * 3u + s) * 2u, pw[j] * dl[q * 256u + c]); miss = miss && c != y; }
    if (miss) { atomicMax(&best[u], j + 1u); }
}
@compute @workgroup_size(256)
fn dense(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    let q = wg.y * 32768u + wg.x;
    if (q >= p.n) { return; }
    for (var l = 0u; l < 2u; l++) {
        let j = q * LV * 4u + l * 4u;
        let r = select(256u + pi[j], pi[j], l == 0u);
        fadd((r * 256u + c) * 2u, pw[j] * dl[q * 256u + c]);
    }
}
