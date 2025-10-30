use divrem::DivCeil;

use crate::discrete_spiking_patch::DiscreteSpikingPatch;
use crate::event::Event;
use crate::token::Tokens;

type Location<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a u16>, ndarray::Dim<[usize; 1]>>;
type Polarity<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a bool>, ndarray::Dim<[usize; 1]>>;
type Time<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a u64>, ndarray::Dim<[usize; 1]>>;

pub struct Tokenizer {
    num_patch_columns: usize,
    patch_size: usize,
    patches: Vec<DiscreteSpikingPatch>,
    spike_threshold: usize,
}

impl Tokenizer {
    pub fn new(
        decay: u64,
        delay: u64,
        height: usize,
        patch_size: usize,
        spike_threshold: usize,
        width: usize,
    ) -> Self {
        let num_patch_rows = DivCeil::div_ceil(height, patch_size);
        let num_patch_columns = DivCeil::div_ceil(width, patch_size);
        let num_patch_columns_u16 = num_patch_columns as u16;
        let num_patches = num_patch_rows * num_patch_columns;
        let mut patches: Vec<DiscreteSpikingPatch> = Vec::with_capacity(num_patches);
        (0..num_patches).for_each(|i| {
            let x = i as u16 % num_patch_columns_u16;
            let y = i as u16 / num_patch_columns_u16;
            patches.push(DiscreteSpikingPatch::new(
                decay,
                delay,
                x,
                y,
                spike_threshold,
            ));
        });

        Tokenizer {
            num_patch_columns,
            patch_size,
            patches,
            spike_threshold,
        }
    }

    pub fn tokenize(&mut self, x: Location, y: Location, t: Time, p: Polarity) -> Tokens {
        let num_events = x.len();
        let max_tokens = num_events / self.spike_threshold;

        let mut token_x: Vec<u16> = Vec::with_capacity(max_tokens);
        let mut token_y: Vec<u16> = Vec::with_capacity(max_tokens);
        let mut token_t: Vec<u64> = Vec::with_capacity(max_tokens);
        let mut events_x: Vec<Vec<u16>> = Vec::with_capacity(max_tokens);
        let mut events_y: Vec<Vec<u16>> = Vec::with_capacity(max_tokens);
        let mut events_t: Vec<Vec<u64>> = Vec::with_capacity(max_tokens);
        let mut events_p: Vec<Vec<bool>> = Vec::with_capacity(max_tokens);

        (0..num_events).for_each(|i| {
            let x = x[i];
            let y = y[i];
            let p = p[i];
            let t = t[i];
            let event = Event { x, y, t, p };

            let patch_x = x as usize / self.patch_size;
            let patch_y = y as usize / self.patch_size;
            let patch_index = patch_y * self.num_patch_columns + patch_x;
            let patch = &mut self.patches[patch_index];

            let token = patch.add(event);
            if let Some(token) = token {
                let (x, y, t, xs, ys, ts, ps) = token;
                token_x.push(x);
                token_y.push(y);
                token_t.push(t);
                events_x.push(xs);
                events_y.push(ys);
                events_t.push(ts);
                events_p.push(ps);
            }
        });

        Tokens {
            x: token_x,
            y: token_y,
            t: token_t,
            events_x,
            events_y,
            events_t,
            events_p,
        }
    }

    pub fn reset(&mut self) {
        self.patches.iter_mut().for_each(|patch| patch.reset());
    }
}
