#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", no_main)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx))]
#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[macro_use]
extern crate derive_more;
// http://prng.di.unimi.it/

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
    phi_phase: *mut f32,
    phi_dist: *mut f32,
    mom_dist: *mut f32,
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
    let phi_phase = from_raw_parts_mut(phi_phase.add(gid * ndet as usize), ndet as usize);
    let len = (ndet * ntof * nmedia) as usize;
    let phi_dist = from_raw_parts_mut(phi_dist.add((gid * len) as usize), len as usize);
    let mom_dist = from_raw_parts_mut(mom_dist.add((gid * len) as usize), len as usize);
    let len = (ndet * ntof) as usize;
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
        phi_phase,
        phi_dist,
        mom_dist,
        photon_counter,
        shared,
    )
}

macro_rules! create_kernel {
    ($kname:ident ; $($src_args:ident : $src_ty:ty $(,)?)* ; $set_src:expr ; $($geom_args:ident : $geom_ty:ty $(,)?)* ; $set_geom:expr) => {
        #[cfg(target_arch = "nvptx64")]
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $kname(
            spec: &MonteCarloSpecification,
            $($src_args : $src_ty,)*
            nmedia: u32,
            states: *const State,
            media: *const u8,
            $($geom_args : $geom_ty,)*
            rngs: *const [u64; 2],
            ndet: u32,
            detectors: *const Detector,
            fluence: *mut f32,
            phi_td: *mut f32,
            phi_phase: *mut f32,
            phi_dist: *mut f32,
            mom_dist: *mut f32,
            photon_counter: *mut u64,
        ) {
            let source = $set_src;
            let geom = $set_geom;
            kernel(
                spec,
                source,
                nmedia,
                states,
                media,
                geom,
                rngs,
                ndet,
                detectors,
                fluence,
                phi_td,
                phi_phase,
                phi_dist,
                mom_dist,
                photon_counter,
            )
        }
    };
}

create_kernel!(pencil ; source: &PencilSource ; source ; geom: &VoxelGeometry ; geom);
create_kernel!(layered_pencil ; source: &PencilSource ; source ; geom_ptr: u64, nlayer: u32 ; core::mem::transmute::<[usize; 2], &LayeredGeometry<VoxelGeometry, [f32]>>([geom_ptr as usize, nlayer as usize]));
create_kernel!(axial_pencil ; source: &PencilSource ; source ; geom: &AxialSymetricGeometry ; geom);
create_kernel!(layered_axial_pencil ; source: &PencilSource ; source ; geom_ptr: u64, nlayer: u32 ; core::mem::transmute::<[usize; 2], &LayeredGeometry<AxialSymetricGeometry, [f32]>>([geom_ptr as usize, nlayer as usize]));
// create_kernel!(free_space_pencil ; source: &PencilSource ; source ;  ;
// &FreeSpaceGeometry);
create_kernel!(layered_free_space_pencil ; source: &PencilSource ; source ; geom_ptr: u64, nlayer: u32 ; core::mem::transmute::<[usize; 2], &LayeredGeometry<FreeSpaceGeometry, [f32]>>([geom_ptr as usize, nlayer as usize]));

create_kernel!(disk ; source: &DiskSource ; source ; geom: &VoxelGeometry ; geom);
create_kernel!(layered_disk ; source: &DiskSource ; source ; geom_ptr: u64, nlayer: u32 ; core::mem::transmute::<[usize; 2], &LayeredGeometry<VoxelGeometry, [f32]>>([geom_ptr as usize, nlayer as usize]));
create_kernel!(axial_disk ; source: &DiskSource ; source ; geom: &AxialSymetricGeometry ; geom);
create_kernel!(layered_axial_disk ; source: &DiskSource ; source ; geom_ptr: u64, nlayer: u32 ; core::mem::transmute::<[usize; 2], &LayeredGeometry<AxialSymetricGeometry, [f32]>>([geom_ptr as usize, nlayer as usize]));
// create_kernel!(free_space_disk ; source: &DiskSource ; source ;  ;
// &FreeSpaceGeometry);
create_kernel!(layered_free_space_disk ; source: &DiskSource ; source ; geom_ptr: u64, nlayer: u32 ; core::mem::transmute::<[usize; 2], &LayeredGeometry<FreeSpaceGeometry, [f32]>>([geom_ptr as usize, nlayer as usize]));

// create_kernel!(pencil layered_pencil axial_pencil layered_axial_pencil
// PencilSource); create_kernel!(pencil_array layered_pencil_array
// [PencilSource]); create_kernel!(disk layered_disk axial_disk
// layered_axial_disk DiskSource); create_kernel!(disk_array layered_disk_array
// [PencilSource]);

#[cfg(not(target_arch = "nvptx64"))]
fn main() {}
