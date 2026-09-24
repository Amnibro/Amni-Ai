@group(0) @binding(0) var<storage, read> pi: array<u32>;
@group(0) @binding(1) var<storage, read> pw: array<f32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> uk: array<u32>;
@group(0) @binding(4) var<storage, read_write> ud: array<u32>;
@group(0) @binding(5) var<storage, read_write> ctr: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = gi(wg, li);
    if (j >= p.n * LV * 4u || pw[j] == 0.0) { return; }
    let l = (j / 4u) % LV;
    let key = l * TE + pi[j];
    if (atomicAdd(&cnt[key], 1u) == 0u) { let u = atomicAdd(&ctr[0], 1u); uk[u] = key; if (l < 2u) { ud[atomicAdd(&ctr[1], 1u)] = u; } }
}
