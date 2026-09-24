@group(0) @binding(0) var lex: texture_2d<u32>;
@group(0) @binding(1) var trg: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> xy: array<u32>;
@group(0) @binding(3) var<storage, read_write> stb: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> pi: array<u32>;
@group(0) @binding(5) var<storage, read_write> pw: array<f32>;
var<private> qb: u32;
fn put(l: u32, k: u32, e: u32, w: f32) { let j = (qb * LV + l) * 4u + k; pi[j] = e; pw[j] = w; }
@compute @workgroup_size(64)
fn state(@builtin(global_invocation_id) g: vec3<u32>) {
    let b = g.x;
    if (b >= p.n) { return; }
    var s: array<u32, 40>;
    s[24] = 1u;
    for (var t = 0u; t < p.t; t++) {
        let q = (b * p.t + t) * 10u;
        upd(&s, xy[b * p.t + t] & 255u);
        stb[q] = vec4<u32>(s[0], s[1], s[2], s[3]); stb[q + 1u] = vec4<u32>(s[4], s[5], s[6], s[7]); stb[q + 2u] = vec4<u32>(s[8], s[9], s[10], s[11]); stb[q + 3u] = vec4<u32>(s[12], s[13], s[14], s[15]); stb[q + 4u] = vec4<u32>(s[16], s[17], s[18], s[19]);
        stb[q + 5u] = vec4<u32>(s[20], s[21], s[22], s[23]); stb[q + 6u] = vec4<u32>(s[24], s[25], s[26], s[27]); stb[q + 7u] = vec4<u32>(s[28], s[29], s[30], s[31]); stb[q + 8u] = vec4<u32>(s[32], s[33], s[34], s[35]); stb[q + 9u] = vec4<u32>(s[36], s[37], s[38], s[39]);
    }
}
@compute @workgroup_size(256)
fn plans(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    qb = gi(wg, li);
    if (qb >= p.n) { return; }
    salt = p.salt;
    var s: array<u32, 40>;
    for (var i = 0u; i < 10u; i++) { let v = stb[qb * 10u + i]; s[i * 4u] = v.x; s[i * 4u + 1u] = v.y; s[i * 4u + 2u] = v.z; s[i * 4u + 3u] = v.w; }
    put(0u, 0u, s[0], 1.0);
    put(1u, 0u, s[1] * 256u + s[0], 1.0);
    plan(&s);
}
