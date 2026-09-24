@group(0) @binding(0) var sidt: texture_storage_2d_array<r32uint, read_write>;
@group(0) @binding(1) var swt: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(2) var<storage, read_write> sm: array<f32>;
@group(0) @binding(3) var<storage, read_write> sv: array<f32>;
@group(0) @binding(4) var<storage, read> ctr: array<u32>;
@group(0) @binding(5) var<storage, read> uk: array<u32>;
@group(0) @binding(6) var<storage, read_write> ga: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> best: array<u32>;
@group(0) @binding(8) var<storage, read> xy: array<u32>;
@group(0) @binding(9) var<storage, read_write> cnt: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let u = gi(wg, li);
    if (u >= ctr[0]) { return; }
    let key = uk[u];
    let l = key / TE;
    if (l < 2u) { return; }
    let e = key % TE;
    let tc = tx(e);
    let o = (l * TE + e) * 3u;
    var w: vec3<f32>;
    var m: vec3<f32>;
    var v: vec3<f32>;
    for (var s = 0u; s < 3u; s++) {
        let g = max(-4.0, min(4.0, fxf(ga[u * 3u + s])));
        ga[u * 3u + s] = vec2<u32>(0u);
        let mm = 0.9 * sm[o + s] + p.o1 * g;
        let vv = 0.99 * sv[o + s] + p.o2 * g * g;
        let ww = textureLoad(swt, tc, l * 3u + s).x * (1.0 - WD * p.lr) - p.lr * (mm / p.c1) / (sqrt(vv / p.c2) + 1e-8);
        let ok = !nan(mm) && !nan(vv) && !nan(ww) && abs(ww) < 1e6;
        m[s] = select(0.0, mm, ok);
        v[s] = select(0.0, vv, ok);
        w[s] = select(0.0, ww, ok);
    }
    let bj = best[u];
    best[u] = 0u;
    if (bj > 0u) {
        var sn = 0u;
        var vn = abs(w[0]);
        for (var s = 1u; s < 3u; s++) { if (abs(w[s]) < vn) { vn = abs(w[s]); sn = s; } }
        if (vn < RTH) {
            let id = textureLoad(sidt, tc, l).x;
            textureStore(sidt, tc, l, vec4<u32>((id & ~(255u << (8u * sn))) | (((xy[(bj - 1u) / (LV * 4u)] >> 8u) & 255u) << (8u * sn)), 0u, 0u, 0u));
            w[sn] = RW;
            m[sn] = 0.0;
            v[sn] = 0.0;
        }
    }
    for (var s = 0u; s < 3u; s++) { textureStore(swt, tc, l * 3u + s, vec4<f32>(w[s], 0.0, 0.0, 0.0)); sm[o + s] = m[s]; sv[o + s] = v[s]; }
    cnt[key] = 0u;
}
