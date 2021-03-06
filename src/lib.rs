#![cfg_attr(not(test), no_std)]
#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate serde;
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
pub use crate::monte_carlo::{monte_carlo, Detector, MonteCarloSpecification, State};
