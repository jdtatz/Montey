use crate::sources::Source;
use crate::utils::*;
use crate::vector::{UnitVector, Vector};
use crate::{
    random::{PRng, UnitCircle},
    Geometry,
};
use rand::{prelude::Distribution, Rng};

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

#[repr(C)]
#[derive(Debug, Copy, Clone, Display)]
#[display(
    fmt = "MonteCarloSpecification(n = {}, Δt = {}, t_max = {}, c = {}, ƒ = {})",
    nphoton,
    dt,
    lifetime_max,
    lightspeed,
    freq
)]
pub struct MonteCarloSpecification {
    pub nphoton: u32,
    pub lifetime_max: f32,
    pub dt: f32,
    pub lightspeed: f32,
    pub freq: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Display)]
#[display(fmt = "State(μ_a = {}, μ_s = {}, g = {}, n = {})", mua, mus, g, n)]
pub struct State {
    pub mua: f32,
    pub mus: f32,
    pub g: f32,
    pub n: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Display)]
#[display(fmt = "Detector(p = {}, r = {})", position, radius)]
pub struct Detector {
    pub position: Vector<f32>,
    pub radius: f32,
}

pub fn monte_carlo<S: Source + ?Sized, G: Geometry + ?Sized>(
    spec: &MonteCarloSpecification,
    src: &S,
    states: &[State],
    media: &[u8],
    geom: &G,
    mut rng: PRng,
    detectors: &[Detector],
    mut fluence: Option<&mut [f32]>,
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
    const PI_2: f32 = 2f32 * core::f32::consts::PI;
    // TODO better name
    let omega_wavelength = PI_2 * spec.freq / spec.lightspeed;

    for _ in 0..spec.nphoton {
        let (mut p, mut v) = src.launch(&mut rng);
        let mut idx = geom.pos2idx(&p);
        let mut weight = 1f32;
        let mut t = 0f32;
        let mut media_id = *fast_index(media, geom.media_index(idx) as usize);
        let mut state = fast_index(states, media_id as usize);
        for opl_mom_j in layer_opl_mom.iter_mut() {
            *opl_mom_j = [0f32; 2];
        }
        let mut ln_phi = 0f32;
        let mut opl = 0f32;

        'photon: loop {
            let rand: f32 = rng.gen();
            let mut mu_t = (state.mua + state.mus).max(1e-12f32);
            let mut s = -rand.ln() / mu_t;
            let (mut dist, mut boundary) = geom.intersection(None, p, v, idx);
            while s > dist {
                p = (*v).mul_add(dist, p);
                t += dist * state.n / spec.lightspeed;
                s -= dist;
                if media_id > 0 {
                    *(&mut fast_index_mut(layer_opl_mom, (media_id - 1) as usize)[0]) +=
                        dist * state.n;
                    ln_phi -= dist * state.mua;
                    opl += dist * state.n;
                }
                let outofbounds = geom.index_step(&mut idx, v, boundary);
                if outofbounds {
                    break 'photon;
                }
                let prev_media_id = core::mem::replace(
                    &mut media_id,
                    *fast_index(media, geom.media_index(idx) as usize),
                );
                if media_id == 0 && prev_media_id != 0 {
                    break 'photon;
                }
                if media_id != prev_media_id {
                    state = fast_index(states, media_id as usize);
                    let prev_mu_t = core::mem::replace(&mut mu_t, state.mua + state.mus);
                    s *= prev_mu_t / mu_t;
                }
                let r = geom.intersection(boundary, p, v, idx);
                dist = r.0;
                boundary = r.1;
            }
            p = (*v).mul_add(s, p);
            t += s * state.n / spec.lightspeed;
            if media_id == 0 || t >= spec.lifetime_max {
                break 'photon;
            }
            *(&mut fast_index_mut(layer_opl_mom, (media_id - 1) as usize)[0]) += s * state.n;
            ln_phi -= s * state.mua;
            opl += s * state.n;
            // absorb
            let delta_weight = weight * state.mua / mu_t;
            if let Some(fluence) = fluence.as_mut() {
                let fidx =
                    geom.fluence_index(idx, (ntof - 1).min((t / spec.dt).floor() as u32), ntof);
                #[cfg(target_arch = "nvptx64")]
                unsafe {
                    let ptr = fluence.as_mut_ptr().add(fidx as usize);
                    nvptx_sys::atomic_load_add_f32(ptr, delta_weight);
                }
                #[cfg(not(target_arch = "nvptx64"))]
                {
                    *fast_index_mut(fluence, fidx as usize) += delta_weight;
                }
            }
            weight -= delta_weight;
            // Scatter
            let rand: f32 = rng.gen();
            let ct = henyey_greenstein_phase(state.g, rand);
            let st = (1f32 - sqr(ct)).sqrt();
            let [cp, sp]: [f32; 2] = UnitCircle.sample(&mut rng);
            v = photon_scatter(&v, ct, st, cp, sp);
            *(&mut fast_index_mut(layer_opl_mom, (media_id - 1) as usize)[1]) += 1f32 - ct;
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
                *fast_index_mut(phi_td, time_id + ntof * i) += phi;
                *fast_index_mut(phi_phase, i) -= phi * opl * omega_wavelength;
                *fast_index_mut(photon_counter, time_id + ntof * i) += 1;
                for (j, [opl_j, mom_j]) in layer_opl_mom.iter().enumerate() {
                    let distr = phi * opl_j / opl;
                    *fast_index_mut(phi_dist, j + nmedia * (time_id + ntof * i)) += distr;
                    *fast_index_mut(mom_dist, j + nmedia * (time_id + ntof * i)) += phi * mom_j;
                }
                break 'detphoton;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PencilSource, VoxelGeometry};
    use ndarray::Array;
    use rand::SeedableRng;
    use std::convert::TryInto;

    #[test]
    fn two_layer() {
        let spec = MonteCarloSpecification {
            nphoton: 100_000,
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
        let geom = VoxelGeometry {
            voxel_dim: Vector::new(1.0, 1.0, 1.0),
            media_dim: Vector::new(200, 200, 200),
        };
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
        let mut media = vec![
            1u8;
            (geom.media_dim.x * geom.media_dim.y * geom.media_dim.z)
                .try_into()
                .unwrap()
        ];
        let depth = 6u32;
        for v in media[((depth * geom.media_dim.y * geom.media_dim.z) as usize)..].iter_mut() {
            *v = 2u8;
        }
        let mut fluence = Array::zeros((
            geom.media_dim.x as usize,
            geom.media_dim.y as usize,
            geom.media_dim.z as usize,
            ntof as usize,
        ));
        let mut phi_td = Array::zeros((ndet as usize, ntof as usize));
        let mut phi_phase = Array::zeros(ndet as usize);
        let mut phi_dist = Array::zeros((ndet as usize, ntof as usize, nlayer as usize));
        let mut mom_dist = Array::zeros((ndet as usize, ntof as usize, nlayer as usize));
        let mut photon_counter = Array::zeros((ndet as usize, ntof as usize));
        let mut layer_opl_mom = vec![[0f32; 2]; nlayer as usize];
        monte_carlo(
            &spec,
            &src,
            &states,
            &media,
            &geom,
            PRng::seed_from_u64(123456u64),
            &dets,
            Some(fluence.as_slice_mut().unwrap()),
            phi_td.as_slice_mut().unwrap(),
            phi_phase.as_slice_mut().unwrap(),
            phi_dist.as_slice_mut().unwrap(),
            mom_dist.as_slice_mut().unwrap(),
            photon_counter.as_slice_mut().unwrap(),
            &mut layer_opl_mom,
        );
        let mut npz = ndarray_npy::NpzWriter::new_compressed(std::io::BufWriter::new(
            std::fs::File::create("test.npz").unwrap(),
        ));
        npz.add_array("fluence", &fluence).unwrap();
        npz.add_array("phi_td", &phi_td).unwrap();
        npz.add_array("phi_phase", &phi_phase).unwrap();
        npz.add_array("phi_dist", &phi_dist).unwrap();
        npz.add_array("mom_dist", &mom_dist).unwrap();
        npz.add_array("photon_counter", &photon_counter).unwrap();
    }
}
