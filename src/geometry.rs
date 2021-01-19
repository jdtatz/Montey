#[cfg(target_arch = "nvptx64")]
use nvptx_sys::Float;
#[cfg(not(target_arch = "nvptx64"))]
use crate::random::Float;
use crate::random::BoolExt;
use crate::vector::{UnitVector, Vector};

pub trait Geometry {
    type Boundary: 'static + Copy + Sized;
    type IdxVector: 'static + Copy + Sized;
    fn pos2idx(&self, pos: &Vector<f32>) -> Self::IdxVector;

    fn media_size(&self) -> usize;
    // media_index
    fn index_3d(&self, index: Self::IdxVector) -> usize;
    // fluence_index
    fn index_4d(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize;

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

    fn index_3d(&self, index: Self::IdxVector) -> usize {
        (index.z + self.media_dim.z * (index.y + self.media_dim.y * index.x)) as usize
    }

    fn index_4d(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize {
        (time_index + time_dim * (index.z + self.media_dim.z * (index.y + self.media_dim.y * index.x))) as usize
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
            _ => (self.voxel_dim.x.min(self.voxel_dim.y).min(self.voxel_dim.z) * 0.5f32, None),
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

impl Geometry for AxialSymetricGeometry {
    type Boundary = ();

    type IdxVector = [u32; 2];

    fn pos2idx(&self, pos: &Vector<f32>) -> Self::IdxVector {
        let [dr, dz] = self.voxel_dim;
        let pos_r = (pos.x * pos.x + pos.y * pos.y).sqrt();
        [
            (pos_r / dr).floor() as u32,
            (pos.z / dz).floor() as u32,
        ]
    }

    fn media_size(&self) -> usize {
        (self.media_dim[0] * self.media_dim[1]) as usize
    }

    fn index_3d(&self, index: Self::IdxVector) -> usize {
        let [_nr, nz] = self.media_dim;
        let [idx_r, idx_z] = index;
        (idx_z + nz * idx_r) as usize
    }

    fn index_4d(&self, index: Self::IdxVector, time_index: u32, time_dim: u32) -> usize {
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
        todo!()
    }

    fn index_step(
        &self,
        idx: &mut Self::IdxVector,
        v: UnitVector<f32>,
        boundary: Option<Self::Boundary>,
    ) -> bool {
        todo!()
    }
}
