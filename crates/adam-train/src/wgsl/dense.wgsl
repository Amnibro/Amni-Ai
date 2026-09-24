@group(0) @binding(0) var dpt: texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var<storage, read_write> dm: array<f32>;
@group(0) @binding(2) var<storage, read_write> dv: array<f32>;
@group(0) @binding(3) var<storage, read> ctr: array<u32>;
@group(0) @binding(4) var<storage, read> ud: array<u32>;
@group(0) @binding(5) var<storage, read> uk: array<u32>;
@group(0) @binding(6) var<storage, read_write> dg: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> cnt: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) c: u32) {
    let d = wg.y * 32768u + wg.x;
    if (d >= ctr[1]) { return; }
    let key = uk[ud[d]];
    let l = key / TE;
    let e = key % TE;
    let a = (select(256u + e, e, l == 0u)) * 256u + c;
    let g = fxf(dg[a]);
    dg[a] = vec2<u32>(0u);
    let r = l * DE + e;
    let i = r * 256u + c;
    let m = 0.9 * dm[i] + p.o1 * g;
    let v = 0.99 * dv[i] + p.o2 * g * g;
    dm[i] = m;
    dv[i] = v;
    let x = textureLoad(dpt, dx(r, c)).x;
    textureStore(dpt, dx(r, c), vec4<f32>(x - p.lr * (m / p.c1) / (sqrt(v / p.c2) + 1e-8), 0.0, 0.0, 0.0));
    if (c == 0u) { cnt[key] = 0u; }
}
