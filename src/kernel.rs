#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", no_main)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx))]
#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate serde;
// http://prng.di.unimi.it/
#[cfg(target_arch = "nvptx64")]
use generate_extern_wrapper::generate_extern_wrapper;

mod random;
mod utils;
pub use crate::random::PRng;
mod vector;
pub use crate::vector::{UnitVector, Vector};
mod sources;
pub use crate::sources::{DiskSource, PencilSource, Source};
mod geometry;
pub use crate::geometry::{AxialSymetricGeometry, FreeSpaceGeometry, Geometry, LayeredGeometry, VoxelGeometry};
mod monte_carlo;
#[cfg(target_arch = "nvptx64")]
use nvptx_sys::{blockDim, blockIdx, threadIdx, Float};

pub use crate::monte_carlo::{monte_carlo, Detector, MonteCarloSpecification, State};

#[cfg(target_arch = "nvptx64")]
#[generate_extern_wrapper(
    format = "{G}{S}",
    abi = "Rust",  // for dst compat, using "Rust" abi instead of "ptx-kernel" abi
    generic(
        param = "G",
        substitute(type = "VoxelGeometry", format = ""),
        substitute(type = "LayeredGeometry::<VoxelGeometry, [f32]>", format = "layered_"),
        substitute(type = "AxialSymetricGeometry", format = "axial_"),
        substitute(type = "LayeredGeometry::<AxialSymetricGeometry, [f32]>", format = "layered_axial_"),
        substitute(type = "LayeredGeometry::<FreeSpaceGeometry, [f32]>", format = "layered_free_space_"),
    ),
    generic(
        param = "S",
        substitute(type = "PencilSource", format = "pencil"),
        substitute(type = "DiskSource", format = "disk"),
        // substitute(type = "[PencilSource]", format = "pencil_array"),
        // substitute(type = "[DiskSource]", format = "disk_array"),
    )
)]
unsafe fn kernel<S: Source + ?Sized, G: Geometry + ?Sized>(
    spec: &MonteCarloSpecification,
    src: &S,
    nmedia: u32,
    states: *const State,
    media: *const u8,
    geom: &G,
    rngs: *const [u64; 2],
    ndet: u32,
    detectors: *const Detector,
    fluence: *mut f32,
    phi_td: *mut f32,
    phi_path_len: *mut f32,
    phi_layer_dist: *mut f32,
    mom_dist: *mut f32,
    photon_weight: *mut f32,
    photon_counter: *mut u64,
) {
    use core::slice::{from_raw_parts, from_raw_parts_mut};

    let ntof = (spec.lifetime_max / spec.dt).ceil() as u32;
    let states = from_raw_parts(states, (nmedia + 1) as usize);
    let media = from_raw_parts(media, geom.media_size());
    let detectors = from_raw_parts(detectors, ndet as usize);
    let fluence = if fluence.is_null() {
        None
    } else {
        Some(from_raw_parts_mut(fluence, geom.fluence_size(ntof)))
    };

    let gid = threadIdx::x() + blockIdx::x() * blockDim::x();
    let rng = core::mem::transmute(rngs.add(gid as usize).read());
    let len = (ndet * ntof) as usize;
    let phi_td = from_raw_parts_mut(phi_td.add((gid * len) as usize), len as usize);
    let phi_path_len = from_raw_parts_mut(phi_path_len.add(gid * ndet as usize), ndet as usize);
    let len = (ndet * ntof * nmedia) as usize;
    let phi_layer_dist = from_raw_parts_mut(phi_layer_dist.add((gid * len) as usize), len as usize);
    let mom_dist = from_raw_parts_mut(mom_dist.add((gid * len) as usize), len as usize);
    let len = (ndet * ntof) as usize;
    let photon_weight = from_raw_parts_mut(photon_weight.add((gid * len) as usize), len as usize);
    let photon_counter = from_raw_parts_mut(photon_counter.add((gid * len) as usize), len as usize);
    let (dyn_mem, dyn_mem_size) = nvptx_sys::dynamic_shared_memory();
    let idx = threadIdx::x() * (nmedia as usize);
    if idx * (nmedia as usize) * core::mem::size_of::<[f32; 2]>() >= dyn_mem_size {
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
    let shared = from_raw_parts_mut((dyn_mem as *mut [f32; 2]).add(idx), nmedia as usize);

    monte_carlo(
        spec,
        src,
        states,
        media,
        geom,
        rng,
        detectors,
        fluence,
        phi_td,
        phi_path_len,
        phi_layer_dist,
        mom_dist,
        photon_weight,
        photon_counter,
        shared,
    )
}

#[cfg(not(target_arch = "nvptx64"))]
fn main() {}
