use crate::event::Event;
use crate::token::Token;

pub struct ContinuousSpikingPatch {
    absolute_refractory_period: u64,
    decay: f64,
    last_time: Option<u64>,
    last_spike_time: Option<u64>,
    patch_x: u16,
    patch_y: u16,
    potential: f64,
    relative_refractory_period: u64,
    relative_refractory_scale: f64,
    spike_threshold: f64,
    events: Vec<Event>,
}

impl ContinuousSpikingPatch {
    pub fn new(
        absolute_refractory_period: u64,
        decay: f64,
        patch_x: u16,
        patch_y: u16,
        relative_refractory_period: u64,
        relative_refractory_scale: f64,
        spike_threshold: f64,
    ) -> Self {
        ContinuousSpikingPatch {
            absolute_refractory_period,
            decay,
            events: Vec::new(),
            last_time: None,
            last_spike_time: None,
            patch_x,
            patch_y,
            potential: 0.0,
            relative_refractory_period,
            relative_refractory_scale,
            spike_threshold,
        }
    }

    pub fn add(&mut self, event: Event) -> Option<Token> {
        let time = event.t;
        if self.in_absolute_refractory_period(time) {
            return None;
        }

        let increase_factor = self.get_increase_scale(time);
        self.update_potential(time, increase_factor);
        self.events.push(event);
        if self.potential >= self.spike_threshold {
            return Some(self.spike(time));
        }
        None
    }

    pub fn reset(&mut self) {
        self.last_time = None;
        self.last_spike_time = None;
        self.potential = 0.0;
        self.events.clear();
    }

    fn update_potential(&mut self, time: u64, increase: f64) {
        let last_time = self.last_time.unwrap_or(time);
        let difference = (time - last_time) as f64;
        self.potential += increase - self.decay * difference;
        self.last_time = Some(time);
        if self.potential <= 0.0 {
            self.potential = 0.0;
            // We deem events as non-informative if the potential decays to 0,
            // and we therefore remove all events from the list
            self.events.clear();
        }
    }

    fn in_absolute_refractory_period(&self, time: u64) -> bool {
        if let Some(last_spike_time) = self.last_spike_time {
            let difference = time - last_spike_time;
            return difference < self.absolute_refractory_period;
        }
        false
    }

    fn get_increase_scale(&self, time: u64) -> f64 {
        if let Some(last_spike_time) = self.last_spike_time {
            let difference = time - last_spike_time;
            if difference < self.relative_refractory_period {
                return self.relative_refractory_scale;
            }
        }
        1.0
    }

    fn spike(&mut self, spike_time: u64) -> Token {
        self.last_time = None;
        self.last_spike_time = Some(spike_time);

        let x = self.patch_x;
        let y = self.patch_y;
        let t = spike_time;

        let events_x = self.events.iter().map(|e| e.x).collect();
        let events_y = self.events.iter().map(|e| e.y).collect();
        let events_t = self.events.iter().map(|e| e.t).collect();
        let events_p = self.events.iter().map(|e| e.p).collect();

        self.events.clear();
        self.potential = 0.0;

        (x, y, t, events_x, events_y, events_t, events_p)
    }
}
