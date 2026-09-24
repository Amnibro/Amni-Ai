@group(0) @binding(0) var<storage, read> ctr: array<u32>;
@group(0) @binding(1) var<storage, read_write> ag: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let w = (ctr[0] + 255u) / 256u;
    let nd = ctr[1];
    ag[0] = min(w, 32768u); ag[1] = (w + 32767u) / 32768u; ag[2] = 1u;
    ag[3] = min(nd, 32768u); ag[4] = (nd + 32767u) / 32768u; ag[5] = 1u;
}
