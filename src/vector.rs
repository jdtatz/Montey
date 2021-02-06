use crate::utils::Float;
use num_traits::{cast, AsPrimitive, NumCast};

#[repr(C)]
#[derive(Debug, Copy, Clone, Neg, Add, Sub, Mul, Div, Display)]
#[display(fmt = "({}, {}, {})", x, y, z)]
pub struct Vector<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector<T> {
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T> Vector<T>
where
    T: Copy + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T>,
{
    pub fn hammard_product(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }

    pub fn dot(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn norm_sqr(self) -> T {
        self.dot(self)
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl Vector<f32> {
    /// self * b + c
    pub fn mul_add(self, b: f32, c: Self) -> Self {
        Self {
            x: self.x.mul_add(b, c.x),
            y: self.y.mul_add(b, c.y),
            z: self.z.mul_add(b, c.z),
        }
    }
}

impl<T: 'static + Copy> Vector<T> {
    pub fn as_<U: 'static + Copy>(self) -> Vector<U>
    where
        T: AsPrimitive<U>,
    {
        Vector::new(self.x.as_(), self.y.as_(), self.z.as_())
    }
}

impl<T: NumCast> Vector<T> {
    pub fn cast<U: NumCast>(self) -> Option<Vector<U>> {
        Some(Vector::new(cast(self.x)?, cast(self.y)?, cast(self.z)?))
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Deref, Into, Display)]
pub struct UnitVector<T>(pub Vector<T>);

impl UnitVector<f32> {
    pub fn new(v: Vector<f32>) -> Option<Self> {
        if (v.norm_sqr() - 1.0).abs() < 1e-8 {
            Some(Self(v))
        } else {
            None
        }
    }
}
