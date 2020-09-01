#![allow(clippy::many_single_char_names, clippy::unreadable_literal, clippy::excessive_precision)]
use num::traits::AsPrimitive;
#[cfg(target_arch = "nvptx64")]
use nvptx_sys::Float;
use rand::distributions::uniform::SampleUniform;
use rand::prelude::{Distribution, Rng};
pub use rand_xoshiro::Xoroshiro128Plus as PRng;

#[cfg(not(target_arch = "nvptx64"))]
pub trait Float: 'static + num::traits::Float {
    const ZERO: Self;
    const ONE: Self;
    fn copysign(self, sign: Self) -> Self;
}

#[cfg(not(target_arch = "nvptx64"))]
impl Float for f32 {
    const ZERO: Self = 0f32;
    const ONE: Self = 1f32;

    fn copysign(self, sign: Self) -> Self {
        libm::copysignf(self, sign)
    }
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

fn polynomial<F: Float + SampleUniform>(z: F, coeff: &[f64]) -> F
where
    f64: AsPrimitive<F>,
{
    let n = coeff.len();
    if n == 0 {
        return F::ZERO;
    }
    let mut sum = coeff[n - 1].as_();
    for i in (0..n - 1).rev() {
        sum = sum.mul_add(z, coeff[i].as_());
    }
    sum
}

const SHAW_P: &[f64] = &[
    1.2533141359896652729,
    3.0333178251950406994,
    2.3884158540184385711,
    0.73176759583280610539,
    0.085838533424158257377,
    0.0034424140686962222423,
    0.000036313870818023761224,
    4.3304513840364031401e-8,
];

const SHAW_Q: &[f64] = &[
    1.0,
    2.9202373175993672857,
    2.9373357991677046357,
    1.2356513216582148689,
    0.2168237095066675527,
    0.014494272424798068406,
    0.00030617264753008793976,
    1.3141263119543315917e-6,
];

/// Fast Non-branching Standard Normal inverse CDF
/// To transform into a normal distribution with stddev=a and mean=b
/// x = b - a * norminv(u)
/// precision is ~1E-9 when 1E-15 <= u <= 1 - 1E-15
/// precision is ~1E-6 when 1E-22 <= u <= 1 - 1E-22
/// precision is ~1E-3 when 1E-30 <= u <= 1 - 1E-30
/// precision is ~1E-2 when 1E-60 <= u <= 1 - 1E-60
/// precision is ~2E-1 when 1E-100 <= u <= 1 - 1E-100
///
/// Source:
/// arXiv:0901.0638 [q-fin.CP]
/// Quantile Mechanics II: Changes of Variables in Monte Carlo methods and GPU-Optimized Normal Quantiles
/// William T. Shaw, Thomas Luu, Nick Brickman
pub fn norminv<F: Float + SampleUniform>(x: F) -> F
where
    f64: AsPrimitive<F>,
{
    let u = if x > (0.5f64).as_() { F::ONE - x } else { x };
    let v = -F::ln((2.0f64).as_() * u);
    let p = polynomial(v, SHAW_P);
    let q = polynomial(v, SHAW_Q);
    (v * p / q).copysign(x - (0.5f64).as_())
}

#[derive(Clone, Copy, Debug)]
struct StandardNormal;

impl<F: Float + SampleUniform> Distribution<F> for StandardNormal
where
    f64: AsPrimitive<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        norminv(rng.gen_range(F::ZERO, F::ONE))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnitCircle;

impl<F: Float + SampleUniform> Distribution<[F; 2]> for UnitCircle
where
    f64: AsPrimitive<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [F; 2] {
        let u: F = StandardNormal.sample(rng);
        let v: F = StandardNormal.sample(rng);
        let d = F::sqrt(u * u + v * v);
        [u / d, v / d]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnitDisc;

impl<F: Float + SampleUniform> Distribution<[F; 2]> for UnitDisc
where
    f64: AsPrimitive<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [F; 2] {
        let [x, y] = UnitCircle.sample(rng);
        let r = F::sqrt(rng.gen_range(F::ZERO, F::ONE));
        [r * x, r * y]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq as assert_almost_eq;
    use statrs::distribution::{InverseCDF, Normal};

    #[test]
    fn test_shaw() {
        let n = Normal::new(0.0, 1.0).unwrap();

        assert_almost_eq!(norminv(1e-100), n.inverse_cdf(1e-100), 2e-1);
        assert_almost_eq!(norminv(1e-60), n.inverse_cdf(1e-60), 2e-2);
        assert_almost_eq!(norminv(1e-30), n.inverse_cdf(1e-30), 1e-3);
        assert_almost_eq!(norminv(1e-20), n.inverse_cdf(1e-20), 1e-5);
        assert_almost_eq!(norminv(1e-15), n.inverse_cdf(1e-15), 5e-9);
        assert_almost_eq!(norminv(1e-10), n.inverse_cdf(1e-10), 5e-9);
        assert_almost_eq!(norminv(1e-5), n.inverse_cdf(1e-5), 2e-9);
        assert_almost_eq!(norminv(0.1), n.inverse_cdf(0.1), 1e-9);
        assert_almost_eq!(norminv(0.2), n.inverse_cdf(0.2), 1e-9);
        assert_almost_eq!(norminv(0.5), n.inverse_cdf(0.5), 1e-9);
        assert_almost_eq!(norminv(0.7), n.inverse_cdf(0.7), 1e-9);
        assert_almost_eq!(norminv(0.9), n.inverse_cdf(0.9), 1e-9);
        assert_almost_eq!(norminv(0.99), n.inverse_cdf(0.99), 2.04e-9);
        assert_almost_eq!(norminv(0.999), n.inverse_cdf(0.999), 2.5e-9);
        assert_almost_eq!(norminv(0.9999), n.inverse_cdf(0.9999), 4e-9);
    }
}
