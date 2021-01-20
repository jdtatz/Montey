#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::excessive_precision
)]
use crate::utils::*;
use num_traits::AsPrimitive;
use rand::distributions::uniform::SampleUniform;
use rand::prelude::{Distribution, Rng};
pub use rand_xoshiro::Xoroshiro128Plus as PRng;

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

#[cfg(test)]
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

#[cfg(test)]
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
/// arXiv:0901.0638v4 [q-fin.CP]
/// Quantile Mechanics II: Changes of Variables in Monte Carlo methods and GPU-Optimized Normal Quantiles
/// William T. Shaw, Thomas Luu, Nick Brickman
#[cfg(test)]
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

const FAST_SHAW_P: &[f64] = &[
    1.2533141012558299407,
    2.4101601285733391215,
    1.3348090307272045436,
    0.23753954196273241709,
    0.011900603295838260268,
    1.1051591117060895699e-4,
];

const FAST_SHAW_Q: &[f64] = &[
    1.0,
    2.4230267574304831865,
    1.8481138350821456213,
    0.50950202270351517687,
    0.046292707412622896113,
    0.0010579909915338770381,
    2.5996479253181457637e-6,
];

/// Fast Non-branching Standard Normal inverse CDF
/// To transform into a normal distribution with stddev=a and mean=b
/// x = b - a * norminv(u)
/// precision is ~2.98E-8 when 2.22E-10 <= u <= 1 - 2.22E-10
/// precision is ~3.1E-8 when 1E-11 <= u <= 1 - 1E-11
/// precision is ~1E-6 when 1E-12 <= u <= 1 - 1E-12
/// precision is ~1E-4 when 1E-17 <= u <= 1 - 1E-16
///
/// Source:
/// arXiv:0901.0638v5 [q-fin.CP]
/// Quantile Mechanics II: Changes of Variables in Monte Carlo methods and GPU-Optimized Normal Quantiles
/// William T. Shaw, Thomas Luu, Nick Brickman
pub fn fast_norminv<F: Float + SampleUniform>(u: F) -> F
where
    f64: AsPrimitive<F>,
{
    let half_minus_u = (0.5f64).as_() - u;
    let mut x = (u * 2f64.as_()).copysign(half_minus_u);
    if half_minus_u < F::ZERO {
        x = x + 2f64.as_();
    }
    let v = -F::ln(x);
    let p = polynomial(v, FAST_SHAW_P);
    let q = polynomial(v, FAST_SHAW_Q);
    (-(p / q)) * v.copysign(half_minus_u)
}

#[derive(Clone, Copy, Debug)]
struct StandardNormal;

impl<F: Float + SampleUniform> Distribution<F> for StandardNormal
where
    f64: AsPrimitive<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        fast_norminv(rng.gen_range(F::ZERO..=F::ONE))
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
        let d = F::rsqrt(u * u + v * v);
        [u * d, v * d]
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
        let r = F::sqrt(rng.gen_range(F::ZERO..=F::ONE));
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

        assert_almost_eq!(norminv(1e-100), n.inverse_cdf(1e-100), max_relative = 2e-1);
        assert_almost_eq!(norminv(1e-60), n.inverse_cdf(1e-60), max_relative = 2e-2);
        assert_almost_eq!(norminv(1e-30), n.inverse_cdf(1e-30), max_relative = 1e-3);
        assert_almost_eq!(norminv(1e-20), n.inverse_cdf(1e-20), max_relative = 1e-5);
        assert_almost_eq!(norminv(1e-15), n.inverse_cdf(1e-15), max_relative = 5e-9);
        assert_almost_eq!(norminv(1e-10), n.inverse_cdf(1e-10), max_relative = 5e-9);
        assert_almost_eq!(norminv(1e-5), n.inverse_cdf(1e-5), max_relative = 2e-9);
        assert_almost_eq!(norminv(0.1), n.inverse_cdf(0.1), max_relative = 1e-9);
        assert_almost_eq!(norminv(0.2), n.inverse_cdf(0.2), max_relative = 1e-9);
        assert_almost_eq!(norminv(0.5), n.inverse_cdf(0.5), max_relative = 1e-9);
        assert_almost_eq!(norminv(0.7), n.inverse_cdf(0.7), max_relative = 1e-9);
        assert_almost_eq!(norminv(0.9), n.inverse_cdf(0.9), max_relative = 1e-9);
        assert_almost_eq!(norminv(0.99), n.inverse_cdf(0.99), max_relative = 2.04e-9);
        assert_almost_eq!(norminv(0.999), n.inverse_cdf(0.999), max_relative = 2.5e-9);
        assert_almost_eq!(norminv(0.9999), n.inverse_cdf(0.9999), max_relative = 4e-9);
    }

    #[test]
    fn test_fast_shaw() {
        let n = Normal::new(0.0, 1.0).unwrap();

        assert_almost_eq!(
            fast_norminv(1e-100),
            n.inverse_cdf(1e-100),
            max_relative = 1e-1
        );
        assert_almost_eq!(
            fast_norminv(1e-60),
            n.inverse_cdf(1e-60),
            max_relative = 1e-2
        );
        assert_almost_eq!(
            fast_norminv(1e-30),
            n.inverse_cdf(1e-30),
            max_relative = 1e-3
        );
        assert_almost_eq!(
            fast_norminv(1e-20),
            n.inverse_cdf(1e-20),
            max_relative = 1e-4
        );
        assert_almost_eq!(
            fast_norminv(1e-15),
            n.inverse_cdf(1e-15),
            max_relative = 5e-5
        );
        assert_almost_eq!(
            fast_norminv(1e-10),
            n.inverse_cdf(1e-10),
            max_relative = 5e-5
        );
        assert_almost_eq!(fast_norminv(1e-5), n.inverse_cdf(1e-5), max_relative = 2e-8);
        assert_almost_eq!(fast_norminv(0.1), n.inverse_cdf(0.1), max_relative = 3e-8);
        assert_almost_eq!(fast_norminv(0.2), n.inverse_cdf(0.2), max_relative = 3e-8);
        assert_almost_eq!(fast_norminv(0.5), n.inverse_cdf(0.5), max_relative = 3e-8);
        assert_almost_eq!(fast_norminv(0.7), n.inverse_cdf(0.7), max_relative = 3e-8);
        assert_almost_eq!(fast_norminv(0.9), n.inverse_cdf(0.9), max_relative = 3e-8);
        assert_almost_eq!(fast_norminv(0.99), n.inverse_cdf(0.99), max_relative = 3e-8);
        assert_almost_eq!(
            fast_norminv(0.999),
            n.inverse_cdf(0.999),
            max_relative = 3e-8
        );
        assert_almost_eq!(
            fast_norminv(0.9999),
            n.inverse_cdf(0.9999),
            max_relative = 4e-8
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-5),
            n.inverse_cdf(1.0 - 1e-5),
            max_relative = 4e-8
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-10),
            n.inverse_cdf(1.0 - 1e-10),
            max_relative = 4e-8
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-11),
            n.inverse_cdf(1.0 - 1e-11),
            max_relative = 3e-7
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-15),
            n.inverse_cdf(1.0 - 1e-15),
            max_relative = 1e-5
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-16),
            n.inverse_cdf(1.0 - 1e-16),
            max_relative = 1e-4
        );
    }
}
