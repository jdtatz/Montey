use core::slice::SliceIndex;

#[cfg(not(target_arch = "nvptx64"))]
#[allow(unused_imports)]
pub(crate) use num_traits::real::Real;
#[cfg(target_arch = "nvptx64")]
pub use nvptx_sys::Float;

pub(crate) fn sqr(x: f32) -> f32 { x * x }

#[cfg(not(target_arch = "nvptx64"))]
pub trait Float: 'static + num_traits::Float + num_traits::NumAssign {
    const ZERO: Self;
    const ONE: Self;
    fn copysign(self, sign: Self) -> Self;
    fn rsqrt(self) -> Self { Self::ONE / self.sqrt() }
}

#[cfg(not(target_arch = "nvptx64"))]
impl Float for f32 {
    const ONE: Self = 1f32;
    const ZERO: Self = 0f32;

    fn copysign(self, sign: Self) -> Self { libm::copysignf(self, sign) }
}

#[cfg(not(target_arch = "nvptx64"))]
impl Float for f64 {
    const ONE: Self = 1f64;
    const ZERO: Self = 0f64;

    fn copysign(self, sign: Self) -> Self { libm::copysign(self, sign) }
}

pub trait BoolExt {
    fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T>;
    fn then_some<T>(self, t: T) -> Option<T>;
    fn if_else<T>(self, true_val: T, false_val: T) -> T;
}

impl BoolExt for bool {
    fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self {
            Some(f())
        } else {
            None
        }
    }

    fn then_some<T>(self, t: T) -> Option<T> {
        if self {
            Some(t)
        } else {
            None
        }
    }

    fn if_else<T>(self, true_val: T, false_val: T) -> T {
        if self {
            true_val
        } else {
            false_val
        }
    }
}

#[macro_export]
macro_rules! fast_unreachable {
    ($message:literal) => {
        #[cfg(target_arch = "nvptx64")]
        unsafe {
            #[cfg(not(debug_assertions))]
            core::hint::unreachable_unchecked();
            #[cfg(debug_assertions)]
            {
                let loc = core::panic::Location::caller();
                nvptx_sys::__assertfail(
                    concat!($message, "\0").as_ptr(),
                    loc.file().as_ptr(),
                    loc.line(),
                    b"\0".as_ptr(),
                    1,
                );
            }
        }
        #[cfg(not(target_arch = "nvptx64"))]
        unreachable!($message)
    };
}

#[track_caller]
pub(crate) fn fast_index<T, I: SliceIndex<[T]>>(slice: &[T], index: I) -> &I::Output {
    if let Some(v) = slice.get(index) {
        v
    } else {
        fast_unreachable!("fast_index out of bounds");
    }
}

#[track_caller]
pub(crate) fn fast_index_mut<T, I: SliceIndex<[T]>>(slice: &mut [T], index: I) -> &mut I::Output {
    if let Some(v) = slice.get_mut(index) {
        v
    } else {
        fast_unreachable!("fast_index_mut out of bounds");
    }
}
