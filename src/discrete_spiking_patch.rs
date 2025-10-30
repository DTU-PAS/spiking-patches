use crate::event::Event;
use crate::token::Token;

pub struct DiscreteSpikingPatch {
    decay: u64,
    delay: u64,
    events: Vec<Event>,
    last_spike_time: Option<u64>,
    patch_x: u16,
    patch_y: u16,
    spike_threshold: usize,
}

impl DiscreteSpikingPatch {
    pub fn new(decay: u64, delay: u64, patch_x: u16, patch_y: u16, spike_threshold: usize) -> Self {
        DiscreteSpikingPatch {
            decay,
            delay,
            events: Vec::with_capacity(spike_threshold),
            patch_x,
            patch_y,
            last_spike_time: None,
            spike_threshold: spike_threshold as usize,
        }
    }

    pub fn add(&mut self, event: Event) -> Option<Token> {
        let time = event.t;

        if self.delay(time) {
            return None;
        }

        self.events.push(event);

        // this first outer check is an optimisation so we don't have to decay on every call
        // the inner check is used to check if the patch has spiked
        if self.events.len() == self.spike_threshold {
            self.decay();

            if self.events.len() == self.spike_threshold {
                return Some(self.spike(time));
            }
        }

        None
    }

    pub fn reset(&mut self) {
        self.last_spike_time = None;
        self.events.clear();
    }

    fn delay(&self, time: u64) -> bool {
        match self.last_spike_time {
            Some(last_spike_time) => time - last_spike_time < self.delay,
            None => false,
        }
    }

    fn decay(&mut self) {
        let last_event = self.events.last().expect("events should not be empty");
        let end_time = last_event.t;
        self.events = self
            .events
            .iter()
            .filter(|event| end_time - event.t <= self.decay)
            .cloned()
            .collect();
    }

    fn spike(&mut self, spike_time: u64) -> Token {
        self.last_spike_time = Some(spike_time);

        let x = self.patch_x;
        let y = self.patch_y;
        let t = spike_time;

        let events_x = self.events.iter().map(|e| e.x).collect();
        let events_y = self.events.iter().map(|e| e.y).collect();
        let events_t = self.events.iter().map(|e| e.t).collect();
        let events_p = self.events.iter().map(|e| e.p).collect();

        self.events.clear();

        (x, y, t, events_x, events_y, events_t, events_p)
    }
}
