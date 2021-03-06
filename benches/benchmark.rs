use std::convert::TryInto;

use criterion::{black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use montey::*;
use ndarray::{s, Array, Axis, Dim, ShapeBuilder, Slice};
use rand::SeedableRng;

struct DisplaySlice<'a, T: std::fmt::Display>(&'a [T]);

impl<'a, T: std::fmt::Display> std::fmt::Display for DisplaySlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for v in self.0.iter() {
            write!(f, "{}, ", v)?;
        }
        write!(f, "]")
    }
}

const SPEC: MonteCarloSpecification = MonteCarloSpecification {
    nphoton:      1_000,
    lifetime_max: 5000.0,
    dt:           100.0,
    lightspeed:   0.2998,
};

const STATES_2_LAYER: &'static [State] = &[
    State {
        mua: 0.0,
        mus: 0.0,
        g:   1.0,
        n:   1.4,
    },
    State {
        mua: 3e-2,
        mus: 10.0,
        g:   0.9,
        n:   1.4,
    },
    State {
        mua: 2e-2,
        mus: 12.0,
        g:   0.9,
        n:   1.4,
    },
];

const VOXEL_SRC: PencilSource = PencilSource {
    src_pos: Vector::new(100.0, 100.0, 0.0),
    src_dir: UnitVector(Vector::new(0.0, 0.0, 1.0)),
};

const VOXEL_GEOM: VoxelGeometry = VoxelGeometry {
    voxel_dim: Vector::new(1.0, 1.0, 1.0),
    media_dim: Vector::new(200, 200, 200),
};

const INSCRIBED_VOXEL_DETECTORS: &'static [Detector] = &[
    Detector {
        position: VOXEL_SRC.src_pos,
        radius:   10.0,
    },
    Detector {
        position: VOXEL_SRC.src_pos,
        radius:   20.0,
    },
    Detector {
        position: VOXEL_SRC.src_pos,
        radius:   30.0,
    },
];

const AXIAL_SYM_SRC: PencilSource = PencilSource {
    src_pos: Vector::new(0.0, 0.0, 0.0),
    src_dir: UnitVector(Vector::new(0.0, 0.0, 1.0)),
};

const AXIAL_SYM_GEOM: AxialSymetricGeometry = AxialSymetricGeometry {
    voxel_dim: [1.0, 1.0],
    media_dim: [200, 200],
};

const INSCRIBED_AXIAL_SYM_DETECTORS: &'static [Detector] = &[
    Detector {
        position: AXIAL_SYM_SRC.src_pos,
        radius:   10.0,
    },
    Detector {
        position: AXIAL_SYM_SRC.src_pos,
        radius:   20.0,
    },
    Detector {
        position: AXIAL_SYM_SRC.src_pos,
        radius:   30.0,
    },
];

fn wrapped_bench_with_input<S: Source + ?Sized, G: Geometry + ?Sized, Sh: Copy + ShapeBuilder>(
    b: &mut Bencher,
    input: &(
        MonteCarloSpecification,
        &S,
        &[State],
        &[u8],
        &G,
        PRng,
        &[Detector],
        Option<Sh>,
    ),
) {
    let (spec, src, states, media, geom, prng, dets, shape) = input;
    let ndet = dets.len();
    let ntof = (spec.lifetime_max / spec.dt).ceil() as usize;
    let nmedia = states.len() - 1;
    let mut fluence = shape.map(|shape| Array::zeros(shape));
    let mut phi_td = Array::zeros((ndet, ntof));
    let mut phi_path_len = Array::zeros(ndet);
    let mut phi_layer_dist = Array::zeros((ndet, ntof, nmedia));
    let mut mom_dist = Array::zeros((ndet, ntof, nmedia));
    let mut photon_weight = Array::zeros((ndet, ntof));
    let mut photon_counter = Array::zeros((ndet, ntof));
    let mut layer_workspace = vec![[0f32; 2]; nmedia];
    b.iter(move || {
        monte_carlo(
            spec,
            *src,
            states,
            media,
            *geom,
            prng.clone(),
            dets,
            fluence.as_mut().map(Array::as_slice_mut).flatten(),
            (phi_td.as_slice_mut().unwrap()),
            (phi_path_len.as_slice_mut().unwrap()),
            (phi_layer_dist.as_slice_mut().unwrap()),
            (mom_dist.as_slice_mut().unwrap()),
            (photon_weight.as_slice_mut().unwrap()),
            (photon_counter.as_slice_mut().unwrap()),
            (&mut layer_workspace),
        )
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    let ntof = (SPEC.lifetime_max / SPEC.dt).ceil() as usize;
    let states = STATES_2_LAYER;
    let depth: isize = 6;
    // Voxel
    let media_dim = [
        VOXEL_GEOM.media_dim.x as usize,
        VOXEL_GEOM.media_dim.y as usize,
        VOXEL_GEOM.media_dim.z as usize,
    ];
    let fluence_dim = [media_dim[0], media_dim[1], media_dim[2], ntof];
    let mut media = Array::ones(media_dim);
    media.slice_mut(s![.., .., depth..]).fill(2u8);
    let input = (
        SPEC,
        &VOXEL_SRC,
        states,
        media.as_slice().unwrap(),
        &VOXEL_GEOM,
        PRng::seed_from_u64(123456u64),
        INSCRIBED_VOXEL_DETECTORS,
        Some(fluence_dim),
    );
    c.bench_with_input(
        BenchmarkId::new("Monte Carlo with Voxel Geometry", DisplaySlice(states)),
        &input,
        wrapped_bench_with_input,
    );
    let lgeom = LayeredGeometry {
        inner_geometry: VOXEL_GEOM,
        layer_bins:     [0f32, 6f32],
    };
    let input = (
        SPEC,
        &VOXEL_SRC,
        states,
        media.as_slice().unwrap(),
        &lgeom as &LayeredGeometry<_, [f32]>,
        PRng::seed_from_u64(123456u64),
        INSCRIBED_VOXEL_DETECTORS,
        Some(fluence_dim),
    );
    c.bench_with_input(
        BenchmarkId::new("Monte Carlo with Layered Voxel Geometry", DisplaySlice(states)),
        &input,
        wrapped_bench_with_input,
    );

    // Axial Symmetric
    let media_dim = [
        AXIAL_SYM_GEOM.media_dim[0] as usize,
        AXIAL_SYM_GEOM.media_dim[1] as usize,
    ];
    let fluence_dim = [media_dim[0], media_dim[1], ntof];
    let mut media = Array::ones(media_dim);
    media.slice_mut(s![.., depth..]).fill(2u8);
    let input = (
        SPEC,
        &AXIAL_SYM_SRC,
        states,
        media.as_slice().unwrap(),
        &AXIAL_SYM_GEOM,
        PRng::seed_from_u64(123456u64),
        INSCRIBED_AXIAL_SYM_DETECTORS,
        Some(fluence_dim),
    );
    c.bench_with_input(
        BenchmarkId::new("Monte Carlo with Axial Symmetric Geometry", DisplaySlice(states)),
        &input,
        wrapped_bench_with_input,
    );
    let lgeom = LayeredGeometry {
        inner_geometry: AXIAL_SYM_GEOM,
        layer_bins:     [0f32, 6f32],
    };
    let input = (
        SPEC,
        &AXIAL_SYM_SRC,
        states,
        media.as_slice().unwrap(),
        &lgeom as &LayeredGeometry<_, [f32]>,
        PRng::seed_from_u64(123456u64),
        INSCRIBED_AXIAL_SYM_DETECTORS,
        Some(fluence_dim),
    );
    c.bench_with_input(
        BenchmarkId::new(
            "Monte Carlo with Layered Axial Symmetric Geometry",
            DisplaySlice(states),
        ),
        &input,
        wrapped_bench_with_input,
    );

    // Free Space
    let media = [0u8, 1, 2];
    let lgeom = LayeredGeometry {
        inner_geometry: FreeSpaceGeometry,
        layer_bins:     [0f32, 6f32],
    };
    let input = (
        SPEC,
        &AXIAL_SYM_SRC,
        states,
        media.as_ref(),
        &lgeom as &LayeredGeometry<_, [f32]>,
        PRng::seed_from_u64(123456u64),
        INSCRIBED_AXIAL_SYM_DETECTORS,
        Option::<[usize; 1]>::None,
    );
    c.bench_with_input(
        BenchmarkId::new("Monte Carlo with Layered Free Space Geometry", DisplaySlice(states)),
        &input,
        wrapped_bench_with_input,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
