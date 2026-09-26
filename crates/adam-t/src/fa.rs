use burn::tensor::{backend::Backend, ops::FloatTensor, DType, Shape, Tensor, TensorData, TensorPrimitive};
use burn_autodiff::{checkpoint::{base::Checkpointer, strategy::CheckpointStrategy}, grads::Gradients, ops::{Backward, Ops, OpsKind}, Autodiff};
use burn_cubecl::{kernel::into_contiguous, ops::numeric::empty_device_dtype, tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_fusion::{stream::{Operation, OperationStreams}, Fusion, FusionBackend, FusionRuntime};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};
use cubecl::{CubeCount, CubeDim};
use std::{marker::PhantomData, sync::LazyLock};
pub static FLASH: LazyLock<bool> = LazyLock::new(|| std::env::var("ATTN").map(|v| v != "ref").unwrap_or(true));
mod kn {
use cubecl::prelude::*;
#[cube(launch)]
pub fn fwd_k(x: &Tensor<f32>, o: &mut Tensor<f32>, lse: &mut Tensor<f32>, t: u32, nh: u32, sc: f32) {
    let (tu, hu, u) = (t as usize, nh as usize, UNIT_POS_X as usize);
    let (bh, qb, r, ln) = (CUBE_POS_Y as usize, CUBE_POS_X as usize, u / 2, u % 2);
    let (b, h, d) = (bh / hu, bh % hu, hu * 64);
    let qi = qb * 64 + r;
    let xb = (b * tu + qi) * 3 * d + h * 64 + ln;
    let mut q = Array::<f32>::new(32usize);
    let mut acc = Array::<f32>::new(32usize);
    let mut s = Array::<f32>::new(8usize);
    #[unroll]
    for j in 0..32usize { q[j] = x[xb + 2 * j] * sc; acc[j] = 0.0; }
    let mut ks = SharedMemory::<f32>::new(4096usize);
    let mut vs = SharedMemory::<f32>::new(4096usize);
    let mut m = f32::new(-1.0e30f32);
    let mut l = f32::new(0.0f32);
    for kb in 0..qb + 1 {
        for e in 0..32usize { let i = e * 128 + u; let src = (b * tu + kb * 64 + i / 64) * 3 * d + h * 64 + i % 64; ks[i] = x[src + d]; vs[i] = x[src + 2 * d]; }
        sync_cube();
        for c0 in 0..8usize {
            let mut mx = m;
            #[unroll]
            for cc in 0..8usize {
                let c = c0 * 8 + cc;
                let mut a = f32::new(0.0f32);
                #[unroll]
                for j in 0..32usize { a += q[j] * ks[c * 64 + 2 * j + ln]; }
                a += plane_shuffle_xor(a, 1u32);
                let v = if kb == qb && c > r { f32::new(-1.0e30f32) } else { a };
                s[cc] = v;
                if v > mx { mx = v; }
            }
            let al = (m - mx).exp();
            l *= al;
            m = mx;
            #[unroll]
            for j in 0..32usize { acc[j] *= al; }
            #[unroll]
            for cc in 0..8usize {
                let p = (s[cc] - m).exp();
                let c = c0 * 8 + cc;
                l += p;
                #[unroll]
                for j in 0..32usize { acc[j] += p * vs[c * 64 + 2 * j + ln]; }
            }
        }
        sync_cube();
    }
    let ob = (b * tu + qi) * d + h * 64 + ln;
    let il = f32::new(1.0f32) / l;
    #[unroll]
    for j in 0..32usize { o[ob + 2 * j] = acc[j] * il; }
    if ln == 0 { lse[bh * tu + qi] = m + l.ln(); }
}
#[cube(launch)]
pub fn dq_k(x: &Tensor<f32>, o: &Tensor<f32>, lse: &Tensor<f32>, g: &Tensor<f32>, dx: &mut Tensor<f32>, dd: &mut Tensor<f32>, t: u32, nh: u32, sc: f32) {
    let (tu, hu, u) = (t as usize, nh as usize, UNIT_POS_X as usize);
    let (bh, qb, r, ln) = (CUBE_POS_Y as usize, CUBE_POS_X as usize, u / 2, u % 2);
    let (b, h, d) = (bh / hu, bh % hu, hu * 64);
    let qi = qb * 64 + r;
    let (xb, ob) = ((b * tu + qi) * 3 * d + h * 64 + ln, (b * tu + qi) * d + h * 64 + ln);
    let mut q = Array::<f32>::new(32usize);
    let mut go = Array::<f32>::new(32usize);
    let mut dq = Array::<f32>::new(32usize);
    let mut di = f32::new(0.0f32);
    #[unroll]
    for j in 0..32usize { q[j] = x[xb + 2 * j] * sc; go[j] = g[ob + 2 * j]; di += go[j] * o[ob + 2 * j]; dq[j] = 0.0; }
    di += plane_shuffle_xor(di, 1u32);
    let li = lse[bh * tu + qi];
    let mut ks = SharedMemory::<f32>::new(4096usize);
    let mut vs = SharedMemory::<f32>::new(4096usize);
    for kb in 0..qb + 1 {
        for e in 0..32usize { let i = e * 128 + u; let src = (b * tu + kb * 64 + i / 64) * 3 * d + h * 64 + i % 64; ks[i] = x[src + d]; vs[i] = x[src + 2 * d]; }
        sync_cube();
        for c in 0..64usize {
            let mut a = f32::new(0.0f32);
            let mut dp = f32::new(0.0f32);
            #[unroll]
            for j in 0..32usize { a += q[j] * ks[c * 64 + 2 * j + ln]; dp += go[j] * vs[c * 64 + 2 * j + ln]; }
            a += plane_shuffle_xor(a, 1u32);
            dp += plane_shuffle_xor(dp, 1u32);
            let p = if kb == qb && c > r { f32::new(0.0f32) } else { (a - li).exp() };
            let ds = p * (dp - di);
            #[unroll]
            for j in 0..32usize { dq[j] += ds * ks[c * 64 + 2 * j + ln]; }
        }
        sync_cube();
    }
    #[unroll]
    for j in 0..32usize { dx[xb + 2 * j] = dq[j] * sc; }
    if ln == 0 { dd[bh * tu + qi] = di; }
}
#[cube(launch)]
pub fn dkv_k(x: &Tensor<f32>, lse: &Tensor<f32>, dd: &Tensor<f32>, g: &Tensor<f32>, dx: &mut Tensor<f32>, t: u32, nh: u32, sc: f32) {
    let (tu, hu, u) = (t as usize, nh as usize, UNIT_POS_X as usize);
    let (bh, kb, r, ln) = (CUBE_POS_Y as usize, CUBE_POS_X as usize, u / 2, u % 2);
    let (b, h, d) = (bh / hu, bh % hu, hu * 64);
    let xb = (b * tu + kb * 64 + r) * 3 * d + h * 64 + ln;
    let mut k = Array::<f32>::new(32usize);
    let mut v = Array::<f32>::new(32usize);
    let mut dk = Array::<f32>::new(32usize);
    let mut dv = Array::<f32>::new(32usize);
    #[unroll]
    for j in 0..32usize { k[j] = x[xb + d + 2 * j]; v[j] = x[xb + 2 * d + 2 * j]; dk[j] = 0.0; dv[j] = 0.0; }
    let mut qs = SharedMemory::<f32>::new(4096usize);
    let mut gs = SharedMemory::<f32>::new(4096usize);
    let mut ls = SharedMemory::<f32>::new(64usize);
    let mut ds = SharedMemory::<f32>::new(64usize);
    for qb in kb..tu / 64 {
        for e in 0..32usize { let i = e * 128 + u; let row = b * tu + qb * 64 + i / 64; qs[i] = x[row * 3 * d + h * 64 + i % 64] * sc; gs[i] = g[row * d + h * 64 + i % 64]; }
        if u < 64 { ls[u] = lse[bh * tu + qb * 64 + u]; ds[u] = dd[bh * tu + qb * 64 + u]; }
        sync_cube();
        for rr in 0..64usize {
            let mut a = f32::new(0.0f32);
            let mut dp = f32::new(0.0f32);
            #[unroll]
            for j in 0..32usize { a += k[j] * qs[rr * 64 + 2 * j + ln]; dp += v[j] * gs[rr * 64 + 2 * j + ln]; }
            a += plane_shuffle_xor(a, 1u32);
            dp += plane_shuffle_xor(dp, 1u32);
            let p = if qb == kb && rr < r { f32::new(0.0f32) } else { (a - ls[rr]).exp() };
            let dsv = p * (dp - ds[rr]);
            #[unroll]
            for j in 0..32usize { dv[j] += p * gs[rr * 64 + 2 * j + ln]; dk[j] += dsv * qs[rr * 64 + 2 * j + ln]; }
        }
        sync_cube();
    }
    #[unroll]
    for j in 0..32usize { dx[xb + d + 2 * j] = dk[j]; dx[xb + 2 * d + 2 * j] = dv[j]; }
}
}
fn ft<B: Backend, const D: usize>(x: FloatTensor<B>) -> Tensor<B, D> { Tensor::from_primitive(TensorPrimitive::Float(x)) }
fn fp<B: Backend, const D: usize>(x: Tensor<B, D>) -> FloatTensor<B> { x.into_primitive().tensor() }
pub fn split<B: Backend>(x: Tensor<B, 2>, b: usize, t: usize, h: usize) -> [Tensor<B, 4>; 3] { let d = x.dims()[1] / 3; let a = x.reshape([b, t, 3 * h, d / h]).swap_dims(1, 2); [a.clone().narrow(1, 0, h), a.clone().narrow(1, h, h), a.narrow(1, 2 * h, h)] }
fn cmask<B: Backend>(t: usize, dev: &B::Device) -> Tensor<B, 4> { Tensor::<B, 2>::from_data(TensorData::new((0..t * t).map(|i| if i % t > i / t { -1.0e30f32 } else { 0.0 }).collect::<Vec<f32>>(), [t, t]), dev).unsqueeze::<4>() }
fn scores<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, t: usize) -> Tensor<B, 4> { let (dh, dev) = (q.dims()[3], q.device()); q.matmul(k.swap_dims(2, 3)).mul_scalar(1.0 / (dh as f64).sqrt()) + cmask::<B>(t, &dev) }
pub fn ref_fwd<B: Backend>(x: Tensor<B, 2>, b: usize, t: usize, h: usize) -> (Tensor<B, 2>, Tensor<B, 1>) {
    let [q, k, v] = split(x, b, t, h);
    let dh = q.dims()[3];
    let s = scores(q, k, t);
    let m = s.clone().max_dim(3);
    let p = (s - m.clone()).exp();
    let l = p.clone().sum_dim(3);
    ((p.matmul(v) / l.clone()).swap_dims(1, 2).reshape([b * t, h * dh]), (m + l.log()).reshape([b * h * t]))
}
pub fn ref_bwd<B: Backend>(x: Tensor<B, 2>, o: Tensor<B, 2>, lse: Tensor<B, 1>, g: Tensor<B, 2>, b: usize, t: usize, h: usize) -> Tensor<B, 2> {
    let [q, k, v] = split(x, b, t, h);
    let dh = q.dims()[3];
    let sc = 1.0 / (dh as f64).sqrt();
    let (o4, g4) = (o.reshape([b, t, h, dh]).swap_dims(1, 2), g.reshape([b, t, h, dh]).swap_dims(1, 2));
    let p = (scores(q.clone(), k.clone(), t) - lse.reshape([b, h, t, 1])).exp();
    let dv = p.clone().swap_dims(2, 3).matmul(g4.clone());
    let dp = g4.clone().matmul(v.swap_dims(2, 3));
    let ds = p * (dp - (g4 * o4).sum_dim(3));
    Tensor::cat(vec![ds.clone().matmul(k).mul_scalar(sc), ds.swap_dims(2, 3).matmul(q).mul_scalar(sc), dv], 1).swap_dims(1, 2).reshape([b * t, 3 * h * dh])
}
pub trait Fa: Backend {
    fn fa_fwd(x: FloatTensor<Self>, b: usize, t: usize, h: usize) -> (FloatTensor<Self>, FloatTensor<Self>) { let (o, l) = ref_fwd::<Self>(ft(x), b, t, h); (fp(o), fp(l)) }
    fn fa_bwd(x: FloatTensor<Self>, o: FloatTensor<Self>, lse: FloatTensor<Self>, g: FloatTensor<Self>, b: usize, t: usize, h: usize) -> FloatTensor<Self> { fp(ref_bwd::<Self>(ft(x), ft(o), ft(lse), ft(g), b, t, h)) }
    fn fa(x: FloatTensor<Self>, b: usize, t: usize, h: usize) -> FloatTensor<Self> { Self::fa_fwd(x, b, t, h).0 }
}
pub fn fa<B: Fa>(x: Tensor<B, 2>, b: usize, t: usize, h: usize) -> Tensor<B, 2> { ft(B::fa(fp(x), b, t, h)) }
impl Fa for burn::backend::NdArray {}
#[cfg(feature = "candle")]
impl Fa for burn::backend::Candle {}
fn ok<R: CubeRuntime>(x: &CubeTensor<R>, t: usize, h: usize) -> bool { x.dtype == DType::F32 && t % 64 == 0 && x.meta.shape()[1] == 3 * 64 * h }
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Fa for CubeBackend<R, F, I, BT> {
    fn fa_fwd(x: FloatTensor<Self>, b: usize, t: usize, h: usize) -> (FloatTensor<Self>, FloatTensor<Self>) {
        if !ok(&x, t, h) { let (o, l) = ref_fwd::<Self>(ft(x), b, t, h); return (fp(o), fp(l)); }
        let x = into_contiguous(x);
        let (c, dev) = (x.client.clone(), x.device.clone());
        let (o, l) = (empty_device_dtype::<R>(c.clone(), dev.clone(), Shape::new([b * t, 64 * h]), DType::F32), empty_device_dtype::<R>(c.clone(), dev, Shape::new([b * h * t]), DType::F32));
        kn::fwd_k::launch::<R>(&c, CubeCount::Static((t / 64) as u32, (b * h) as u32, 1), CubeDim::new_1d(128), x.into_tensor_arg(), o.clone().into_tensor_arg(), l.clone().into_tensor_arg(), t as u32, h as u32, 0.125f32);
        (o, l)
    }
    fn fa_bwd(x: FloatTensor<Self>, o: FloatTensor<Self>, lse: FloatTensor<Self>, g: FloatTensor<Self>, b: usize, t: usize, h: usize) -> FloatTensor<Self> {
        if !ok(&x, t, h) { return fp(ref_bwd::<Self>(ft(x), ft(o), ft(lse), ft(g), b, t, h)); }
        let (x, o, lse, g) = (into_contiguous(x), into_contiguous(o), into_contiguous(lse), into_contiguous(g));
        let (c, dev) = (x.client.clone(), x.device.clone());
        let (dx, dd) = (empty_device_dtype::<R>(c.clone(), dev.clone(), Shape::new([b * t, 3 * 64 * h]), DType::F32), empty_device_dtype::<R>(c.clone(), dev, Shape::new([b * h * t]), DType::F32));
        let (cc, cd) = (CubeCount::Static((t / 64) as u32, (b * h) as u32, 1), CubeDim::new_1d(128));
        kn::dq_k::launch::<R>(&c, cc.clone(), cd, x.clone().into_tensor_arg(), o.into_tensor_arg(), lse.clone().into_tensor_arg(), g.clone().into_tensor_arg(), dx.clone().into_tensor_arg(), dd.clone().into_tensor_arg(), t as u32, h as u32, 0.125f32);
        kn::dkv_k::launch::<R>(&c, cc, cd, x.into_tensor_arg(), lse.into_tensor_arg(), dd.into_tensor_arg(), g.into_tensor_arg(), dx.clone().into_tensor_arg(), t as u32, h as u32, 0.125f32);
        dx
    }
}
#[derive(Clone, Debug)]
struct FwdOp<B> { desc: CustomOpIr, s: [usize; 3], _b: PhantomData<B> }
#[derive(Clone, Debug)]
struct BwdOp<B> { desc: CustomOpIr, s: [usize; 3], _b: PhantomData<B> }
impl<B: FusionBackend + Fa> Operation<B::FusionRuntime> for FwdOp<B> {
    fn execute(&self, hs: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>) { let ([x], [o, l]) = self.desc.as_fixed(); let [b, t, h] = self.s; let (ro, rl) = B::fa_fwd(hs.get_float_tensor::<B>(x), b, t, h); hs.register_float_tensor::<B>(&o.id, ro); hs.register_float_tensor::<B>(&l.id, rl); }
}
impl<B: FusionBackend + Fa> Operation<B::FusionRuntime> for BwdOp<B> {
    fn execute(&self, hs: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>) { let ([x, o, l, g], [dx]) = self.desc.as_fixed(); let [b, t, h] = self.s; let r = B::fa_bwd(hs.get_float_tensor::<B>(x), hs.get_float_tensor::<B>(o), hs.get_float_tensor::<B>(l), hs.get_float_tensor::<B>(g), b, t, h); hs.register_float_tensor::<B>(&dx.id, r); }
}
impl<B: FusionBackend + Fa> Fa for Fusion<B> {
    fn fa_fwd(x: FloatTensor<Self>, b: usize, t: usize, h: usize) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (c, dt, d) = (x.client.clone(), x.dtype, x.shape[1] / 3);
        let st = OperationStreams::with_inputs([&x]);
        let (o, l) = (TensorIr::uninit(c.create_empty_handle(), Shape::new([b * t, d]), dt), TensorIr::uninit(c.create_empty_handle(), Shape::new([b * h * t]), dt));
        let desc = CustomOpIr::new("adam_t_fa_fwd", &[x.into_ir()], &[o, l]);
        let [o, l]: [FloatTensor<Self>; 2] = c.register(st, OperationIr::Custom(desc.clone()), FwdOp::<B> { desc, s: [b, t, h], _b: PhantomData }).try_into().unwrap();
        (o, l)
    }
    fn fa_bwd(x: FloatTensor<Self>, o: FloatTensor<Self>, lse: FloatTensor<Self>, g: FloatTensor<Self>, b: usize, t: usize, h: usize) -> FloatTensor<Self> {
        let (c, dt, sh) = (x.client.clone(), x.dtype, x.shape.clone());
        let st = OperationStreams::with_inputs([&x, &o, &lse, &g]);
        let dx = TensorIr::uninit(c.create_empty_handle(), sh, dt);
        let desc = CustomOpIr::new("adam_t_fa_bwd", &[x.into_ir(), o.into_ir(), lse.into_ir(), g.into_ir()], &[dx]);
        c.register(st, OperationIr::Custom(desc.clone()), BwdOp::<B> { desc, s: [b, t, h], _b: PhantomData }).remove(0)
    }
}
#[derive(Debug)]
struct FaBack;
impl<B: Fa> Backward<B, 1> for FaBack {
    type State = (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, [usize; 3]);
    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let [p] = ops.parents;
        let g = grads.consume::<B>(&ops.node);
        let (x, o, l, [b, t, h]) = ops.state;
        if let Some(n) = p { grads.register::<B>(n.id, B::fa_bwd(x, o, l, g, b, t, h)); }
    }
}
impl<B: Fa, C: CheckpointStrategy> Fa for Autodiff<B, C> {
    fn fa(x: FloatTensor<Self>, b: usize, t: usize, h: usize) -> FloatTensor<Self> {
        match FaBack.prepare::<C>([x.node.clone()]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => { let (o, l) = B::fa_fwd(x.primitive.clone(), b, t, h); prep.finish((x.primitive, o.clone(), l, [b, t, h]), o) }
            OpsKind::UnTracked(prep) => prep.finish(B::fa_fwd(x.primitive, b, t, h).0),
        }
    }
}
