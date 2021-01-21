use crate::vector::{UnitVector, Vector};
use crate::{fast_unreachable, utils::*};

pub trait Geometry {
    type Boundary: 'static + Copy + Sized;
    type IdxVector: 'static + Copy + Sized;
    fn pos2idx(&self, pos: &Vector<f32>) -> Self::IdxVector;

    fn media_size(&self) -> usize;
    fn fluence_size(&self, time_dim: u32) -> usize;
    fn media_index(&self, index: Self::IdxVector) -> usize;
    fn fluence_index(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize;

    fn intersection(
        &self,
        prev: Option<Self::Boundary>,
        pos: Vector<f32>,
        v: UnitVector<f32>,
        idx: Self::IdxVector,
    ) -> (f32, Option<Self::Boundary>);

    fn index_step(
        &self,
        idx: &mut Self::IdxVector,
        v: UnitVector<f32>,
        boundary: Option<Self::Boundary>,
    ) -> bool;
}

pub struct FreeSpaceGeometry;

impl Geometry for FreeSpaceGeometry {
    type Boundary = ();
    type IdxVector = ();
    fn pos2idx(&self, _pos: &Vector<f32>) -> Self::IdxVector {
        ()
    }

    fn media_size(&self) -> usize {
        0
    }

    fn fluence_size(&self, _time_dim: u32) -> usize {
        fast_unreachable!("FreeSpaceGeometry does not support fluence");
    }

    fn media_index(&self, _index: Self::IdxVector) -> usize {
        0
    }

    fn fluence_index(&self, _index: Self::IdxVector, _time_index: u32, _time_dim: u32) -> usize {
        fast_unreachable!("FreeSpaceGeometry does not support fluence");
    }

    fn intersection(
        &self,
        _prev: Option<Self::Boundary>,
        _pos: Vector<f32>,
        _v: UnitVector<f32>,
        _idx: Self::IdxVector,
    ) -> (f32, Option<Self::Boundary>) {
        (core::f32::INFINITY, None)
    }

    fn index_step(
        &self,
        _idx: &mut Self::IdxVector,
        _v: UnitVector<f32>,
        _boundary: Option<Self::Boundary>,
    ) -> bool {
        false
    }
}

pub struct VoxelGeometry {
    pub voxel_dim: Vector<f32>,
    pub media_dim: Vector<u32>,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum VoxelBoundary {
    X,
    Y,
    Z,
}

impl Geometry for VoxelGeometry {
    type Boundary = VoxelBoundary;

    type IdxVector = Vector<u32>;

    fn pos2idx(&self, pos: &Vector<f32>) -> Self::IdxVector {
        Vector {
            x: (pos.x / self.voxel_dim.x).floor() as u32,
            y: (pos.y / self.voxel_dim.y).floor() as u32,
            z: (pos.z / self.voxel_dim.z).floor() as u32,
        }
    }

    fn media_size(&self) -> usize {
        (self.media_dim.x * self.media_dim.y * self.media_dim.z) as usize
    }

    fn fluence_size(&self, time_dim: u32) -> usize {
        (self.media_dim.x * self.media_dim.y * self.media_dim.z * time_dim) as usize
    }

    fn media_index(&self, index: Self::IdxVector) -> usize {
        (index.z + self.media_dim.z * (index.y + self.media_dim.y * index.x)) as usize
    }

    fn fluence_index(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize {
        (time_index
            + time_dim * (index.z + self.media_dim.z * (index.y + self.media_dim.y * index.x)))
            as usize
    }

    fn intersection(
        &self,
        prev: Option<Self::Boundary>,
        pos: Vector<f32>,
        v: UnitVector<f32>,
        idx: Self::IdxVector,
    ) -> (f32, Option<Self::Boundary>) {
        let voxel_pos = pos - self.voxel_dim.hammard_product(idx.as_());
        let dx = matches!(prev, Some(VoxelBoundary::X)).if_else(
            self.voxel_dim.x,
            v.x.is_sign_positive()
                .if_else(self.voxel_dim.x - voxel_pos.x, voxel_pos.x),
        );
        let dy = matches!(prev, Some(VoxelBoundary::Y)).if_else(
            self.voxel_dim.y,
            v.y.is_sign_positive()
                .if_else(self.voxel_dim.y - voxel_pos.y, voxel_pos.y),
        );
        let dz = matches!(prev, Some(VoxelBoundary::Z)).if_else(
            self.voxel_dim.z,
            v.z.is_sign_positive()
                .if_else(self.voxel_dim.z - voxel_pos.z, voxel_pos.z),
        );

        let hx = BoolExt::then(v.x != 0f32, || (dx / v.x).abs());
        let hy = BoolExt::then(v.y != 0f32, || (dy / v.y).abs());
        let hz = BoolExt::then(v.z != 0f32, || (dz / v.z).abs());

        match (hx, hy, hz) {
            (Some(x), Some(y), Some(z)) if x <= y && x <= z => (x, Some(VoxelBoundary::X)),
            (Some(x), Some(y), None) if x <= y => (x, Some(VoxelBoundary::X)),
            (Some(x), None, Some(z)) if x <= z => (x, Some(VoxelBoundary::X)),
            (Some(x), None, None) => (x, Some(VoxelBoundary::X)),
            (_, Some(y), Some(z)) if y <= z => (y, Some(VoxelBoundary::Y)),
            (_, Some(y), None) => (y, Some(VoxelBoundary::Y)),
            (_, _, Some(z)) => (z, Some(VoxelBoundary::Z)),
            // TODO: choose a more correct way of handling odd starts
            _ => (
                self.voxel_dim.x.min(self.voxel_dim.y).min(self.voxel_dim.z) * 0.5f32,
                None,
            ),
        }
    }

    fn index_step(
        &self,
        idx: &mut Self::IdxVector,
        v: UnitVector<f32>,
        boundary: Option<Self::Boundary>,
    ) -> bool {
        match boundary {
            Some(VoxelBoundary::X) if v.x.is_sign_positive() => {
                idx.x += 1;
                idx.x >= self.media_dim.x
            }
            Some(VoxelBoundary::X) if v.x.is_sign_negative() => {
                if idx.x > 0 {
                    idx.x -= 1;
                    false
                } else {
                    true
                }
            }
            Some(VoxelBoundary::Y) if v.y.is_sign_positive() => {
                idx.y += 1;
                idx.y >= self.media_dim.y
            }
            Some(VoxelBoundary::Y) if v.y.is_sign_negative() => {
                if idx.y > 0 {
                    idx.y -= 1;
                    false
                } else {
                    true
                }
            }
            Some(VoxelBoundary::Z) if v.z.is_sign_positive() => {
                idx.z += 1;
                idx.z >= self.media_dim.z
            }
            Some(VoxelBoundary::Z) if v.z.is_sign_negative() => {
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
}

pub struct AxialSymetricGeometry {
    pub voxel_dim: [f32; 2],
    pub media_dim: [u32; 2],
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum AxialSymetricBoundary {
    Radial(bool),
    Axial,
}

impl Geometry for AxialSymetricGeometry {
    type Boundary = AxialSymetricBoundary;

    type IdxVector = [u32; 2];

    fn pos2idx(&self, pos: &Vector<f32>) -> Self::IdxVector {
        let [dr, dz] = self.voxel_dim;
        let pos_r = (pos.x * pos.x + pos.y * pos.y).sqrt();
        [(pos_r / dr).floor() as u32, (pos.z / dz).floor() as u32]
    }

    fn media_size(&self) -> usize {
        (self.media_dim[0] * self.media_dim[1]) as usize
    }

    fn fluence_size(&self, time_dim: u32) -> usize {
        (self.media_dim[0] * self.media_dim[1] * time_dim) as usize
    }

    fn media_index(&self, index: Self::IdxVector) -> usize {
        let [_nr, nz] = self.media_dim;
        let [idx_r, idx_z] = index;
        (idx_z + nz * idx_r) as usize
    }

    fn fluence_index(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize {
        let [_nr, nz] = self.media_dim;
        let [idx_r, idx_z] = index;
        (time_index + time_dim * (idx_z + nz * idx_r)) as usize
    }

    fn intersection(
        &self,
        prev: Option<Self::Boundary>,
        pos: Vector<f32>,
        v: UnitVector<f32>,
        idx: Self::IdxVector,
    ) -> (f32, Option<Self::Boundary>) {
        let pos_r = (pos.x * pos.x + pos.y * pos.y).sqrt();
        let v_r = (1f32 - v.z * v.z)
            .sqrt()
            .copysign(v.x * v.x + v.y * v.y + 2f32 * pos.x * v.x + 2f32 * pos.y * v.y);
        let voxel_r = pos_r - self.voxel_dim[0] * (idx[0] as f32);
        let voxel_z = pos.z - self.voxel_dim[1] * (idx[1] as f32);
        let dr = matches!(prev, Some(AxialSymetricBoundary::Radial(_))).if_else(
            self.voxel_dim[0],
            v_r.is_sign_positive()
                .if_else(self.voxel_dim[0] - voxel_r, voxel_r),
        );
        let dz = matches!(prev, Some(AxialSymetricBoundary::Axial)).if_else(
            self.voxel_dim[1],
            v.z.is_sign_positive()
                .if_else(self.voxel_dim[1] - voxel_z, voxel_z),
        );

        let hr = BoolExt::then(v_r != 0f32, || (dr / v_r).abs());
        let hz = BoolExt::then(v.z != 0f32, || (dz / v.z).abs());

        match (hr, hz) {
            (Some(r), Some(z)) if r <= z => (
                r,
                Some(AxialSymetricBoundary::Radial(v_r.is_sign_positive())),
            ),
            (Some(r), None) => (
                r,
                Some(AxialSymetricBoundary::Radial(v_r.is_sign_positive())),
            ),
            (_, Some(z)) => (z, Some(AxialSymetricBoundary::Axial)),
            // TODO: choose a more correct way of handling odd starts
            _ => (self.voxel_dim[0].min(self.voxel_dim[1]) * 0.5f32, None),
        }
    }

    fn index_step(
        &self,
        idx: &mut Self::IdxVector,
        v: UnitVector<f32>,
        boundary: Option<Self::Boundary>,
    ) -> bool {
        match boundary {
            Some(AxialSymetricBoundary::Radial(true)) => {
                idx[0] += 1;
                idx[0] >= self.media_dim[0]
            }
            Some(AxialSymetricBoundary::Radial(false)) => {
                if idx[0] > 0 {
                    idx[0] -= 1;
                }
                false
            }
            Some(AxialSymetricBoundary::Axial) if v.z.is_sign_positive() => {
                idx[1] += 1;
                idx[1] >= self.media_dim[1]
            }
            Some(AxialSymetricBoundary::Axial) => {
                if idx[1] > 0 {
                    idx[1] -= 1;
                    false
                } else {
                    true
                }
            }
            None => false,
        }
    }
}

pub struct LayeredGeometry<G: Geometry> {
    pub inner_geometry: G,
    pub layer_bins: [f32],
}

#[derive(Clone, Copy, Debug)]
pub enum LayeredBoundary<B> {
    Inner(B),
    Layer,
}

impl<B> LayeredBoundary<B> {
    fn into_inner(self) -> Option<B> {
        match self {
            Self::Inner(b) => Some(b),
            Self::Layer => None,
        }
    }
}

impl<G: Geometry> Geometry for LayeredGeometry<G> {
    type Boundary = LayeredBoundary<G::Boundary>;

    type IdxVector = (G::IdxVector, u32);

    fn pos2idx(&self, pos: &Vector<f32>) -> Self::IdxVector {
        let idx_0 = self.inner_geometry.pos2idx(pos);
        let mut idx_1 = 0;
        for &b in self.layer_bins.iter() {
            if pos.z < b {
                return (idx_0, idx_1);
            }
            idx_1 += 1;
        }
        (idx_0, idx_1 + 1)
    }

    fn media_size(&self) -> usize {
        self.layer_bins.len() + 1
    }

    fn fluence_size(&self, time_dim: u32) -> usize {
        self.inner_geometry.fluence_size(time_dim)
    }

    fn media_index(&self, index: Self::IdxVector) -> usize {
        index.1 as usize
    }

    fn fluence_index(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize {
        self.inner_geometry
            .fluence_index(index.0, time_index, time_dim)
    }

    fn intersection(
        &self,
        prev: Option<Self::Boundary>,
        pos: Vector<f32>,
        v: UnitVector<f32>,
        idx: Self::IdxVector,
    ) -> (f32, Option<Self::Boundary>) {
        let (dist_0, bound) = self.inner_geometry.intersection(
            prev.map(LayeredBoundary::into_inner).flatten(),
            pos,
            v,
            idx.0,
        );
        let dist_1 = if v.z.is_sign_positive() && idx.1 < (self.layer_bins.len() as u32) {
            fast_index(&self.layer_bins, idx.1 as usize) - pos.z
        } else if v.z.is_sign_negative() && idx.1 > 0 {
            pos.z - fast_index(&self.layer_bins, (idx.1 - 1) as usize)
        } else {
            // core::f32::INFINITY
            return (dist_0, bound.map(LayeredBoundary::Inner));
        };
        if dist_0 <= dist_1 {
            (dist_0, bound.map(LayeredBoundary::Inner))
        } else {
            (dist_1, Some(LayeredBoundary::Layer))
        }
    }

    fn index_step(
        &self,
        idx: &mut Self::IdxVector,
        v: UnitVector<f32>,
        boundary: Option<Self::Boundary>,
    ) -> bool {
        match boundary {
            Some(LayeredBoundary::Inner(b)) => {
                self.inner_geometry.index_step(&mut idx.0, v, Some(b))
            }
            Some(LayeredBoundary::Layer) if v.z.is_sign_positive() => {
                idx.1 += 1;
                false
            }
            Some(LayeredBoundary::Layer) => {
                idx.1 -= 1;
                false
            }
            None => false,
        }
    }
}
