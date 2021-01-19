#[cfg(not(target_arch = "nvptx64"))]
use crate::random::Float;
use crate::random::{BoolExt, PRng, UnitCircle};
use crate::sources::Source;
use crate::vector::{UnitVector, Vector};
use core::slice::SliceIndex;
#[cfg(target_arch = "nvptx64")]
use nvptx_sys::Float;
use rand::{prelude::Distribution, Rng};

fn sqr(x: f32) -> f32 {
    x * x
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
enum Boundary {
    X,
    Y,
    Z,
}

fn intersection(
    prev: Option<Boundary>,
    voxel_pos: Vector<f32>,
    v: UnitVector<f32>,
    voxel_dim: Vector<f32>,
) -> (f32, Option<Boundary>) {
    let dx = matches!(prev, Some(Boundary::X)).if_else(
        voxel_dim.x,
        v.x.is_sign_positive()
            .if_else(voxel_dim.x - voxel_pos.x, voxel_pos.x),
    );
    let dy = matches!(prev, Some(Boundary::Y)).if_else(
        voxel_dim.y,
        v.y.is_sign_positive()
            .if_else(voxel_dim.y - voxel_pos.y, voxel_pos.y),
    );
    let dz = matches!(prev, Some(Boundary::Z)).if_else(
        voxel_dim.z,
        v.z.is_sign_positive()
            .if_else(voxel_dim.z - voxel_pos.z, voxel_pos.z),
    );

    let hx = BoolExt::then(v.x != 0f32, || (dx / v.x).abs());
    let hy = BoolExt::then(v.y != 0f32, || (dy / v.y).abs());
    let hz = BoolExt::then(v.z != 0f32, || (dz / v.z).abs());

    match (hx, hy, hz) {
        (Some(x), Some(y), Some(z)) if x <= y && x <= z => (x, Some(Boundary::X)),
        (Some(x), Some(y), None) if x <= y => (x, Some(Boundary::X)),
        (Some(x), None, Some(z)) if x <= z => (x, Some(Boundary::X)),
        (Some(x), None, None) => (x, Some(Boundary::X)),
        (_, Some(y), Some(z)) if y <= z => (y, Some(Boundary::Y)),
        (_, Some(y), None) => (y, Some(Boundary::Y)),
        (_, _, Some(z)) => (z, Some(Boundary::Z)),
        // TODO: choose a more correct way of handling odd starts
        _ => (voxel_dim.x.min(voxel_dim.y).min(voxel_dim.z) * 0.5f32, None),
    }
}

fn henyey_greenstein_phase(g: f32, rand: f32) -> f32 {
    if g != 0f32 {
        let g2 = sqr(g);
        (1f32 / (2f32 * g)) * (1f32 + g2 - sqr((1f32 - g2) / (1f32 - g + 2f32 * g * rand)))
    } else {
        1f32 - 2f32 * rand
    }
}

fn photon_scatter(v: &Vector<f32>, ct: f32, st: f32, cp: f32, sp: f32) -> UnitVector<f32> {
    const EPS_N_1: f32 = 1f32 - 1e-6f32;
    if v.z.abs() < EPS_N_1 {
        let d = 1f32 - sqr(v.z);
        let denom = d.sqrt();
        let rdenom = d.rsqrt();
        UnitVector(Vector {
            x: st * (v.x * v.z * cp - v.y * sp) * rdenom + v.x * ct,
            y: st * (v.y * v.z * cp + v.x * sp) * rdenom + v.y * ct,
            z: -denom * st * cp + v.z * ct,
        })
    } else {
        UnitVector(Vector {
            x: st * cp,
            y: st * sp * (1f32).copysign(v.z),
            z: ct * (1f32).copysign(v.z),
        })
    }
}

fn index_step(
    idx: &mut Vector<u32>,
    v: UnitVector<f32>,
    media_dim: Vector<u32>,
    boundary: Option<Boundary>,
) -> bool {
    match boundary {
        Some(Boundary::X) if v.x.is_sign_positive() => {
            idx.x += 1;
            idx.x >= media_dim.x
        }
        Some(Boundary::X) if v.x.is_sign_negative() => {
            if idx.x > 0 {
                idx.x -= 1;
                false
            } else {
                true
            }
        }
        Some(Boundary::Y) if v.y.is_sign_positive() => {
            idx.y += 1;
            idx.y >= media_dim.y
        }
        Some(Boundary::Y) if v.y.is_sign_negative() => {
            if idx.y > 0 {
                idx.y -= 1;
                false
            } else {
                true
            }
        }
        Some(Boundary::Z) if v.z.is_sign_positive() => {
            idx.z += 1;
            idx.z >= media_dim.z
        }
        Some(Boundary::Z) if v.z.is_sign_negative() => {
            if idx.z > 0 {
                idx.z -= 1;
                false
            } else {
                true
            }
        }
        _ => false,
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MonteCarloSpecification {
    pub nphoton: u32,
    pub voxel_dim: Vector<f32>,
    pub lifetime_max: f32,
    pub dt: f32,
    pub lightspeed: f32,
    pub freq: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct State {
    pub mua: f32,
    pub mus: f32,
    pub g: f32,
    pub n: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Detector {
    pub position: Vector<f32>,
    pub radius: f32,
}

fn index_3d(index: Vector<u32>, dim: Vector<u32>) -> u32 {
    index.z + dim.z * (index.y + dim.y * index.x)
}

fn index_3d_time(index: Vector<u32>, time_index: u32, dim: Vector<u32>, time_dim: u32) -> u32 {
    time_index + time_dim * (index.z + dim.z * (index.y + dim.y * index.x))
}

fn pos2idx(pos: &Vector<f32>, voxel_dim: &Vector<f32>) -> Vector<u32> {
    Vector {
        x: (pos.x / voxel_dim.x).floor() as u32,
        y: (pos.y / voxel_dim.y).floor() as u32,
        z: (pos.z / voxel_dim.z).floor() as u32,
    }
}

#[track_caller]
fn safe_index<T, I: SliceIndex<[T]>>(slice: &[T], index: I) -> &I::Output {
    if let Some(v) = slice.get(index) {
        v
    } else {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            #[cfg(not(debug_assertions))]
            core::hint::unreachable_unchecked();
            #[cfg(debug_assertions)]
            {
                let loc = core::panic::Location::caller();
                nvptx_sys::__assertfail(
                    b"safe_index out of bounds\0".as_ptr(),
                    loc.file().as_ptr(),
                    loc.line(),
                    b"\0".as_ptr(),
                    1,
                );
            }
        }
        #[cfg(not(target_arch = "nvptx64"))]
        unreachable!("safe_index out of bounds")
    }
}

#[track_caller]
fn safe_index_mut<T, I: SliceIndex<[T]>>(slice: &mut [T], index: I) -> &mut I::Output {
    if let Some(v) = slice.get_mut(index) {
        v
    } else {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            #[cfg(not(debug_assertions))]
            core::hint::unreachable_unchecked();
            #[cfg(debug_assertions)]
            {
                let loc = core::panic::Location::caller();
                nvptx_sys::__assertfail(
                    b"safe_index_mut out of bounds\0".as_ptr(),
                    loc.file().as_ptr(),
                    loc.line(),
                    b"\0".as_ptr(),
                    1,
                );
            }
        }
        #[cfg(not(target_arch = "nvptx64"))]
        unreachable!("safe_index_mut out of bounds")
    }
}

pub fn monte_carlo<S: Source + ?Sized>(
    spec: &MonteCarloSpecification,
    src: &S,
    states: &[State],
    media_dim: Vector<u32>,
    media: &[u8],
    mut rng: PRng,
    detectors: &[Detector],
    fluence: &mut [f32],
    phi_td: &mut [f32],
    phi_phase: &mut [f32],
    phi_dist: &mut [f32],
    mom_dist: &mut [f32],
    photon_counter: &mut [u64],
    layer_opl_mom: &mut [[f32; 2]],
) {
    let ntof = (spec.lifetime_max / spec.dt).ceil() as u32;
    // let ndet = detectors.len();
    let nmedia = states.len() - 1;
    let media_size = spec.voxel_dim.hammard_product(media_dim.into());
    const PI_2: f32 = 2f32 * core::f32::consts::PI;
    // TODO better name
    let omega_wavelength = PI_2 * spec.freq / spec.lightspeed;

    for _ in 0..spec.nphoton {
        let (mut p, mut v) = src.launch(&mut rng);
        debug_assert!(0f32 <= p.x && p.x < media_size.x);
        debug_assert!(0f32 <= p.y && p.y < media_size.y);
        debug_assert!(0f32 <= p.z && p.z < media_size.z);
        let mut idx = pos2idx(&p, &spec.voxel_dim);
        let mut weight = 1f32;
        let mut t = 0f32;
        let mut media_id = *safe_index(media, index_3d(idx, media_dim) as usize);
        let mut state = safe_index(states, media_id as usize);
        for opl_mom_j in layer_opl_mom.iter_mut() {
            *opl_mom_j = [0f32; 2];
        }
        let mut ln_phi = 0f32;
        let mut opl = 0f32;

        'photon: loop {
            let rand: f32 = rng.gen();
            let mut mu_t = (state.mua + state.mus).max(1e-12f32);
            let mut s = -rand.ln() / mu_t;
            let voxel_pos = p - spec.voxel_dim.hammard_product(idx.into());
            let (mut dist, mut boundary) = intersection(None, voxel_pos, v, spec.voxel_dim);
            while s > dist {
                p = (*v).mul_add(dist, p);
                t += dist * state.n / spec.lightspeed;
                s -= dist;
                if media_id > 0 {
                    *(&mut safe_index_mut(layer_opl_mom, (media_id - 1) as usize)[0]) +=
                        dist * state.n;
                    ln_phi -= dist * state.mua;
                    opl += dist * state.n;
                }
                let outofbounds = index_step(&mut idx, v, media_dim, boundary);
                if outofbounds {
                    break 'photon;
                }
                let prev_media_id = core::mem::replace(
                    &mut media_id,
                    *safe_index(media, index_3d(idx, media_dim) as usize),
                );
                if media_id == 0 && prev_media_id != 0 {
                    break 'photon;
                }
                if media_id != prev_media_id {
                    state = safe_index(states, media_id as usize);
                    let prev_mu_t = core::mem::replace(&mut mu_t, state.mua + state.mus);
                    s *= prev_mu_t / mu_t;
                }
                let voxel_pos = p - spec.voxel_dim.hammard_product(idx.into());
                let r = intersection(boundary, voxel_pos, v, spec.voxel_dim);
                dist = r.0;
                boundary = r.1;
            }
            p = (*v).mul_add(s, p);
            t += s * state.n / spec.lightspeed;
            if media_id == 0 || t >= spec.lifetime_max {
                break 'photon;
            }
            *(&mut safe_index_mut(layer_opl_mom, (media_id - 1) as usize)[0]) += s * state.n;
            ln_phi -= s * state.mua;
            opl += s * state.n;
            // absorb
            let delta_weight = weight * state.mua / mu_t;
            let fidx = index_3d_time(idx, (ntof - 1).min((t / spec.dt).floor() as u32), media_dim, ntof);
            #[cfg(target_arch = "nvptx64")]
            unsafe {
                let ptr = fluence.as_mut_ptr().add(fidx as usize);
                nvptx_sys::atomic_load_add_f32(ptr, delta_weight);
            }
            #[cfg(not(target_arch = "nvptx64"))]
            {
                *safe_index_mut(fluence, fidx as usize) += delta_weight;
            }

            weight -= delta_weight;
            // Scatter
            let rand: f32 = rng.gen();
            let ct = henyey_greenstein_phase(state.g, rand);
            let st = (1f32 - sqr(ct)).sqrt();
            let [cp, sp]: [f32; 2] = UnitCircle.sample(&mut rng);
            v = photon_scatter(&v, ct, st, cp, sp);
            *(&mut safe_index_mut(layer_opl_mom, (media_id - 1) as usize)[1]) += 1f32 - ct;
            // roulette
            const ROULETTE_THRESHOLD: f32 = 1e-4f32;
            const ROULETTE_CONSTANT: f32 = 10f32;
            const ROULETTE_CONSTANT_RECIP: f32 = 1f32 / ROULETTE_CONSTANT;
            if weight < ROULETTE_THRESHOLD {
                let rand: f32 = rng.gen();
                if rand > ROULETTE_CONSTANT_RECIP {
                    break 'photon;
                }
                weight *= ROULETTE_CONSTANT;
            }
        }
        // detected photons?
        'detphoton: for (i, det) in detectors.iter().enumerate() {
            let sqr_dist = (det.position - p).norm_sqr();
            if sqr_dist < sqr(det.radius) {
                let ntof = ntof as usize;
                let time_id = (ntof - 1).min((t / spec.dt).floor() as usize);
                let phi = ln_phi.exp();
                *safe_index_mut(phi_td, time_id + ntof * i) += phi;
                *safe_index_mut(phi_phase, i) -= phi * opl * omega_wavelength;
                *safe_index_mut(photon_counter, time_id + ntof * i) += 1;
                for (j, [opl_j, mom_j]) in layer_opl_mom.iter().enumerate() {
                    let distr = phi * opl_j / opl;
                    *safe_index_mut(phi_dist, j + nmedia * (time_id + ntof * i)) += distr;
                    *safe_index_mut(mom_dist, j + nmedia * (time_id + ntof * i)) += phi * mom_j;
                }
                break 'detphoton;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;
    use rand::SeedableRng;
    use crate::PencilSource;
    use super::*;
    use ndarray::Array;

    #[test]
    fn two_layer() {
        let spec = MonteCarloSpecification {
            nphoton: 100_000,
            voxel_dim: Vector::new(1.0, 1.0, 1.0),
            lifetime_max: 5000.0,
            dt: 100.0,
            lightspeed: 0.2998,
            freq: 110e-6,
        };
        let ntof = (spec.lifetime_max / spec.dt).ceil() as u32;
        let src = PencilSource {
            src_pos: Vector::new(0.0, 100.0, 100.0),
            src_dir: UnitVector::new(Vector::new(1.0, 0.0, 0.0)).unwrap(),
        };
        let states = [
            State {
                mua: 0.0,
                mus: 0.0,
                g: 1.0,
                n: 1.4,
            },
            State {
                mua: 3e-2,
                mus: 10.0,
                g: 0.9,
                n: 1.4,
            },
            State {
                mua: 2e-2,
                mus: 12.0,
                g: 0.9,
                n: 1.4,
            },
        ];
        let nlayer = states.len() as u32 - 1;
        let dets = [
            Detector {
                position: src.src_pos.clone(),
                radius: 10.0,
            },
            Detector {
                position: src.src_pos.clone(),
                radius: 20.0,
            },
            Detector {
                position: src.src_pos.clone(),
                radius: 30.0,
            },
        ];
        let ndet = dets.len() as u32;
        let media_dim = Vector::new(200u32, 200, 200);
        let mut media = vec![1u8; (media_dim.x * media_dim.y * media_dim.z).try_into().unwrap()];
        let depth = 6u32;
        for v in media[((depth * media_dim.y * media_dim.z) as usize)..].iter_mut() {
            *v = 2u8;
        }
        let mut fluence = Array::zeros(
            (media_dim.x as usize, media_dim.y as usize, media_dim.z as usize, ntof as usize)
        );
        let mut phi_td  = Array::zeros(
            (ndet as usize, ntof as usize)
        );
        let mut phi_phase= Array::zeros(
            ndet as usize
        );
        let mut phi_dist= Array::zeros(
            (ndet as usize, ntof as usize, nlayer as usize)
        );
        let mut mom_dist= Array::zeros(
            (ndet as usize, ntof as usize, nlayer as usize)
        );
        let mut photon_counter= Array::zeros(
            (ndet as usize, ntof as usize)
        );
        let mut layer_opl_mom = vec![[0f32; 2]; nlayer as usize];
        monte_carlo(
            &spec,
            &src,
            &states,
            media_dim,
            &media,
            PRng::seed_from_u64(123456u64),
            &dets,
            fluence.as_slice_mut().unwrap(),
            phi_td.as_slice_mut().unwrap(),
            phi_phase.as_slice_mut().unwrap(),
            phi_dist.as_slice_mut().unwrap(),
            mom_dist.as_slice_mut().unwrap(),
            photon_counter.as_slice_mut().unwrap(),
            &mut layer_opl_mom,
        );
        let mut npz = ndarray_npy::NpzWriter::new_compressed(std::io::BufWriter::new(std::fs::File::create("test.npz").unwrap()));
        npz.add_array("fluence", &fluence).unwrap();
        npz.add_array("phi_td", &phi_td).unwrap();
        npz.add_array("phi_phase", &phi_phase).unwrap();
        npz.add_array("phi_dist", &phi_dist).unwrap();
        npz.add_array("mom_dist", &mom_dist).unwrap();
        npz.add_array("photon_counter", &photon_counter).unwrap();
    }
}
