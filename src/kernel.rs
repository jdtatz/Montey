#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", no_main)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx))]
#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[macro_use]
extern crate derive_more;
// http://prng.di.unimi.it/

mod random;
pub use crate::random::PRng;
mod vector;
pub use crate::vector::{UnitVector, Vector};
mod sources;
pub use crate::sources::{DiskSource, PencilSource, Source};
mod monte_carlo;
pub use crate::monte_carlo::{monte_carlo, Detector, MonteCarloSpecification, State};

#[cfg(target_arch = "nvptx64")]
use nvptx_sys::{blockDim, blockIdx, threadIdx, Float};

#[cfg(target_arch = "nvptx64")]
unsafe fn kernel<S: Source + ?Sized>(
    spec: &MonteCarloSpecification,
    src: &S,
    nmedia: u32,
    states: *const State,
    nx: u32,
    ny: u32,
    nz: u32,
    media: *const u8,
    rngs: *const [u64; 2],
    ndet: u32,
    detectors: *const Detector,
    fluence: *mut f32,
    phi_td: *mut f32,
    phi_phase: *mut f32,
    phi_dist: *mut f32,
    photon_counter: *mut u64,
) {
    let ntof = (spec.lifetime_max / spec.dt).ceil() as u32;

    let states = core::slice::from_raw_parts(states, (nmedia + 1) as usize);
    let media_dim = Vector::new(nx, ny, nz);
    let media = core::slice::from_raw_parts(media, (nx * ny * nz) as usize);
    let detectors = core::slice::from_raw_parts(detectors, ndet as usize);
    let fluence = core::slice::from_raw_parts_mut(fluence, (nx * ny * nz * ntof) as usize);

    let gid = threadIdx::x() + blockIdx::x() * blockDim::x();
    let rng = core::mem::transmute(rngs.add(gid as usize).read());
    let len = (ndet * ntof) as usize;
    let phi_td = core::slice::from_raw_parts_mut(phi_td.add((gid * len) as usize), len as usize);
    let phi_phase =
        core::slice::from_raw_parts_mut(phi_phase.add(gid * ndet as usize), ndet as usize);
    let len = (ndet * ntof * nmedia) as usize;
    let phi_dist =
        core::slice::from_raw_parts_mut(phi_dist.add((gid * len) as usize), len as usize);
    let len = (ndet * ntof) as usize;
    let photon_counter =
        core::slice::from_raw_parts_mut(photon_counter.add((gid * len) as usize), len as usize);
    let (dyn_mem, dyn_mem_size) = nvptx_sys::dynamic_shared_memory();
    let idx = threadIdx::x() * (nmedia as usize);
    if idx * (nmedia as usize) * core::mem::size_of::<f32>() >= dyn_mem_size {
        #[cfg(not(debug_assertions))]
        core::hint::unreachable_unchecked();
        #[cfg(debug_assertions)]
        {
            nvptx_sys::__assertfail(
                b"Not enough dynamic shared memory allocated\0".as_ptr(),
                concat!(file!(), "\0").as_ptr(),
                line!(),
                b"\0".as_ptr(),
                1,
            );
        }
    }
    let shared = core::slice::from_raw_parts_mut((dyn_mem as *mut f32).add(idx), nmedia as usize);

    monte_carlo(
        spec,
        src,
        states,
        media_dim,
        media,
        rng,
        detectors,
        fluence,
        phi_td,
        phi_phase,
        phi_dist,
        photon_counter,
        shared,
    )
}

macro_rules! create_kernel {
    ($kname:ident $karname:ident $src:ty) => {
        #[cfg(target_arch = "nvptx64")]
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $kname(
            spec: &MonteCarloSpecification,
            source: &$src,
            nmedia: u32,
            states: *const State,
            nx: u32,
            ny: u32,
            nz: u32,
            media: *const u8,
            rngs: *const [u64; 2],
            ndet: u32,
            detectors: *const Detector,
            fluence: *mut f32,
            phi_td: *mut f32,
            phi_phase: *mut f32,
            phi_dist: *mut f32,
            photon_counter: *mut u64,
        ) {
            kernel(
                spec,
                source,
                nmedia,
                states,
                nx,
                ny,
                nz,
                media,
                rngs,
                ndet,
                detectors,
                fluence,
                phi_td,
                phi_phase,
                phi_dist,
                photon_counter,
            )
        }

        #[cfg(target_arch = "nvptx64")]
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $karname(
            spec: &MonteCarloSpecification,
            nsources: u32,
            sources: *const $src,
            nmedia: u32,
            states: *const State,
            nx: u32,
            ny: u32,
            nz: u32,
            media: *const u8,
            rngs: *const [u64; 2],
            ndet: u32,
            detectors: *const Detector,
            fluence: *mut f32,
            phi_td: *mut f32,
            phi_phase: *mut f32,
            phi_dist: *mut f32,
            photon_counter: *mut u64,
        ) {
            kernel(
                spec,
                core::slice::from_raw_parts(sources, nsources as usize),
                nmedia,
                states,
                nx,
                ny,
                nz,
                media,
                rngs,
                ndet,
                detectors,
                fluence,
                phi_td,
                phi_phase,
                phi_dist,
                photon_counter,
            )
        }
    };
}

create_kernel!(pencil pencil_array PencilSource);
create_kernel!(disk disk_array DiskSource);

#[cfg(not(target_arch = "nvptx64"))]
fn main() {}
