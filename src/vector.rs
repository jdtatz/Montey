#[cfg(target_arch = "nvptx64")]
use nvptx_sys::Float;

#[repr(C)]
#[derive(Debug, Copy, Clone, Neg, Add, Sub, Mul, Div)]
pub struct Vector<T> {
    pub(crate) x: T,
    pub(crate) y: T,
    pub(crate) z: T,
}

impl<T> Vector<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl Vector<f32> {
    pub fn hammard_product(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn norm_sqr(self) -> f32 {
        self.dot(self)
    }

    /// self * b + c
    pub fn mul_add(self, b: f32, c: Self) -> Self {
        Self {
            x: self.x.mul_add(b, c.x),
            y: self.y.mul_add(b, c.y),
            z: self.z.mul_add(b, c.z),
        }
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl From<Vector<u32>> for Vector<f32> {
    fn from(v: Vector<u32>) -> Self {
        Vector {
            x: v.x as f32,
            y: v.y as f32,
            z: v.z as f32,
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Deref, Into)]
pub struct UnitVector<T>(pub(crate) Vector<T>);

impl UnitVector<f32> {
    pub fn new(v: Vector<f32>) -> Option<Self> {
        if (v.norm_sqr() - 1.0).abs() < 1e-8 {
            Some(Self(v))
        } else {
            None
        }
    }
}
