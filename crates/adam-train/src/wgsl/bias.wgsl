@group(0) @binding(0) var<storage, read> dl: array<f32>;
@group(0) @binding(1) var<storage, read_write> bp: array<f32>;
@group(0) @binding(2) var bit: texture_storage_2d<r32float, read_write>;
@compute @workgroup_size(256)
fn part(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    var z = 0.0;
    for (var q = wg.x * 256u; q < min(p.n, wg.x * 256u + 256u); q++) { z += dl[q * 256u + c]; }
    bp[wg.x * 256u + c] = z;
}
@compute @workgroup_size(256)
fn fin(@builtin(local_invocation_index) c: u32) {
    var z = 0.0;
    for (var k = 0u; k < (p.n + 255u) / 256u; k++) { z += bp[k * 256u + c]; }
    let b = textureLoad(bit, vec2<u32>(c, 0u)).x;
    textureStore(bit, vec2<u32>(c, 0u), vec4<f32>(b - p.lr * 0.1 * z, 0.0, 0.0, 0.0));
}
