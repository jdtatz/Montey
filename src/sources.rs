use rand::{prelude::Distribution, seq::SliceRandom};

use crate::{fast_unreachable, random::UnitDisc, utils::*, PRng, UnitVector, Vector};

pub trait Source {
    fn launch(&self, rng: &mut PRng) -> (Vector<f32>, UnitVector<f32>);
}

impl<S: Source> Source for &S {
    fn launch(&self, rng: &mut PRng) -> (Vector<f32>, UnitVector<f32>) { S::launch(*self, rng) }
}

impl<S: Source> Source for [S] {
    fn launch(&self, rng: &mut PRng) -> (Vector<f32>, UnitVector<f32>) {
        // can't use seq::choose b/c bounds check isn't optimized out
        // have to use, instead of rng.gen_range(0, self.len())
        // rng.gen::<f32>() => [0, 1)
        if let Some(src) = self.choose(rng) {
            src.launch(rng)
        } else {
            fast_unreachable!("No elements in slice");
        }
    }
}

impl<S: Source, const N: usize> Source for [S; N] {
    fn launch(&self, rng: &mut PRng) -> (Vector<f32>, UnitVector<f32>) {
        // can't use seq::choose b/c bounds check isn't optimized out
        // have to use, instead of rng.gen_range(0, self.len())
        // rng.gen::<f32>() => [0, 1)
        if let Some(src) = self.choose(rng) {
            src.launch(rng)
        } else {
            fast_unreachable!("No elements in slice");
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PencilSource {
    pub src_pos: Vector<f32>,
    pub src_dir: UnitVector<f32>,
}

impl Source for PencilSource {
    fn launch(&self, _rng: &mut PRng) -> (Vector<f32>, UnitVector<f32>) { (self.src_pos, self.src_dir) }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DiskSource {
    pub src_pos:           Vector<f32>,
    pub src_dir:           UnitVector<f32>,
    pub orthonormal_basis: [UnitVector<f32>; 2],
    pub radius:            f32,
}

impl DiskSource {
    pub fn new(src_pos: Vector<f32>, src_dir: UnitVector<f32>, radius: f32) -> Self {
        let z = UnitVector(Vector { x: 0.0, y: 0.0, z: 1.0 });
        let x_vec = *z - *src_dir * src_dir.dot(*z);
        let y_vec = src_dir.cross(x_vec);
        Self {
            src_pos,
            src_dir,
            orthonormal_basis: [UnitVector(x_vec), UnitVector(y_vec)],
            radius,
        }
    }
}

impl Source for DiskSource {
    fn launch(&self, rng: &mut PRng) -> (Vector<f32>, UnitVector<f32>) {
        // pre-generate orthonormal_basis
        #[cfg(test)]
        {
            let z = UnitVector(Vector { x: 0.0, y: 0.0, z: 1.0 });
            let x_vec = *z - *self.src_dir * self.src_dir.dot(*z);
            let y_vec = self.src_dir.cross(x_vec);
            approx::assert_ulps_eq!(x_vec.dot(*self.orthonormal_basis[0]), 1.);
            approx::assert_ulps_eq!(y_vec.dot(*self.orthonormal_basis[1]), 1.);
        }
        #[cfg(test)]
        {
            let [x_vec, y_vec] = self.orthonormal_basis;
            approx::assert_ulps_eq!(self.src_dir.dot(*x_vec), 0.);
            approx::assert_ulps_eq!(self.src_dir.dot(*y_vec), 0.);
            approx::assert_ulps_eq!(x_vec.dot(*y_vec), 0.);
        }
        let [x_vec, y_vec] = self.orthonormal_basis;
        // Uses rejection method, might want to use Muller-Marsaglia with fast sin & cos
        let [u, v]: [f32; 2] = UnitDisc.sample(rng);
        let x = u * self.radius;
        let y = v * self.radius;
        let p = self.src_pos + *x_vec * x + *y_vec * y;
        (p, self.src_dir)
    }
}

// TODO Add Source tests
#[cfg(test)]
mod tests {}
