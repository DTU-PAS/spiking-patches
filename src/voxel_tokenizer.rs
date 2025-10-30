use divrem::DivCeil;
use ndarray::Zip;

use crate::token::Tokens;

type Location<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a u16>, ndarray::Dim<[usize; 1]>>;
type Polarity<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a bool>, ndarray::Dim<[usize; 1]>>;
type Time<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a u64>, ndarray::Dim<[usize; 1]>>;

pub struct Tokenizer {
    duration_us: u64,
    height: usize,
    patch_size: usize,
    threshold: usize,
    width: usize,
}

impl Tokenizer {
    pub fn new(
        duration_us: u64,
        height: usize,
        patch_size: usize,
        threshold: usize,
        width: usize,
    ) -> Self {
        Tokenizer {
            duration_us,
            height,
            patch_size,
            threshold,
            width,
        }
    }

    pub fn tokenize(&self, x: Location, y: Location, t: Time, p: Polarity) -> Tokens {
        // initialize voxels
        let min_time = *t.first().unwrap_or(&0);
        let max_time = *t.last().unwrap_or(&0);
        let bin_start = min_time / self.duration_us;
        let bin_start_time = bin_start * self.duration_us;
        let bin_end = DivCeil::div_ceil(max_time, self.duration_us);
        let num_time_bins = bin_end - bin_start + 1;
        let num_rows = DivCeil::div_ceil(self.height, self.patch_size);
        let num_cols = DivCeil::div_ceil(self.width, self.patch_size);
        let area = num_rows * num_cols;
        let num_voxels = (num_time_bins as usize) * area;
        let mut voxels: Vec<Voxel> = Vec::with_capacity(num_voxels);
        for time_bin in 0..num_time_bins {
            let time = bin_start_time + time_bin * self.duration_us;
            for row in 0..num_rows {
                for col in 0..num_cols {
                    let voxel = Voxel::new(col as u16, row as u16, time as u64);
                    voxels.push(voxel);
                }
            }
        }

        // add events to voxels
        Zip::from(x)
            .and(y)
            .and(t)
            .and(p)
            .for_each(|&x, &y, &t, &p| {
                let time_bin = ((t / self.duration_us) - bin_start) as usize;
                let row = (y as usize) / self.patch_size;
                let col = (x as usize) / self.patch_size;
                let voxel_index = time_bin * area + row * num_cols + col;
                let voxel = &mut voxels[voxel_index];
                voxel.add(x, y, t, p);
            });

        // filter voxels based on threshold
        voxels.retain(|v| v.events_x.len() >= self.threshold);

        let pos_x: Vec<u16> = voxels.iter().map(|v| v.x).collect();
        let pos_y: Vec<u16> = voxels.iter().map(|v| v.y).collect();
        let pos_t: Vec<u64> = voxels.iter().map(|v| v.t).collect();
        let events_x = voxels.iter().map(|v| v.events_x.clone()).collect();
        let events_y = voxels.iter().map(|v| v.events_y.clone()).collect();
        let events_t = voxels.iter().map(|v| v.events_t.clone()).collect();
        let events_p = voxels.iter().map(|v| v.events_p.clone()).collect();

        Tokens {
            x: pos_x,
            y: pos_y,
            t: pos_t,
            events_x,
            events_y,
            events_t,
            events_p,
        }
    }
}

struct Voxel {
    x: u16,
    y: u16,
    t: u64,
    events_x: Vec<u16>,
    events_y: Vec<u16>,
    events_t: Vec<u64>,
    events_p: Vec<bool>,
}

impl Voxel {
    fn new(x: u16, y: u16, t: u64) -> Self {
        Voxel {
            x,
            y,
            t,
            events_x: Vec::new(),
            events_y: Vec::new(),
            events_t: Vec::new(),
            events_p: Vec::new(),
        }
    }

    fn add(&mut self, x: u16, y: u16, t: u64, p: bool) {
        self.events_x.push(x);
        self.events_y.push(y);
        self.events_t.push(t);
        self.events_p.push(p);
    }
}
