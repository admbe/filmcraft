//! How long the UI's own work takes per frame (`logic` + `ui`), for `perf.stats.ui` and the
//! slow-frame log. The fps reading is an average over frames and hides a single long one; a long
//! frame is exactly what a user sees as a freeze (an import probing every clip, a source opened on
//! the UI thread), so each one over [`SLOW_MS`] is logged with the half of the frame it was in.

use std::time::Duration;
use web_time::Instant;

/// A frame this long is logged as a warning.
pub const SLOW_MS: f64 = 250.0;

/// The two halves of an eframe frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    /// `App::logic`: control requests, commands, playback and preroll bookkeeping.
    Logic,
    /// `App::ui`: the panels (timeline, monitors, project panel…).
    Panels,
}

/// Per-frame timing; cumulative counters, read with [`UiTiming::stats`].
#[derive(Debug, Default)]
pub struct UiTiming {
    started: Option<Instant>,
    logic: Duration,
    panels: Duration,
    frames: u64,
    over_50: u64,
    over_100: u64,
    over_250: u64,
    longest_ms: f64,
    slow_logged: u64,
}

impl UiTiming {
    /// A half of the frame starts.
    pub fn begin(&mut self) {
        self.started = Some(Instant::now());
    }

    /// The half that [`begin`](Self::begin) started ends. Ending [`Half::Panels`] closes the frame.
    pub fn end(&mut self, half: Half) {
        let Some(t0) = self.started.take() else { return };
        let took = t0.elapsed();
        match half {
            Half::Logic => self.logic = self.logic.saturating_add(took),
            Half::Panels => {
                self.panels = self.panels.saturating_add(took);
                self.close_frame();
            }
        }
    }

    fn close_frame(&mut self) {
        let (logic, panels) = (std::mem::take(&mut self.logic), std::mem::take(&mut self.panels));
        let ms = logic.saturating_add(panels).as_secs_f64() * 1e3;
        self.record(ms, logic.as_secs_f64() * 1e3, panels.as_secs_f64() * 1e3);
    }

    /// One frame of `ms` (`logic_ms` + `panels_ms`).
    fn record(&mut self, ms: f64, logic_ms: f64, panels_ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        self.frames = self.frames.saturating_add(1);
        self.over_50 += u64::from(ms > 50.0);
        self.over_100 += u64::from(ms > 100.0);
        self.over_250 += u64::from(ms > SLOW_MS);
        self.longest_ms = self.longest_ms.max(ms);
        if ms > SLOW_MS {
            self.slow_logged = self.slow_logged.saturating_add(1);
            let half = if logic_ms >= panels_ms { "logic (commands, control requests, playback)" } else { "panels" };
            log::warn!("slow UI frame: {ms:.0} ms ({half}: logic {logic_ms:.0} ms, panels {panels_ms:.0} ms); the window did not respond meanwhile");
        }
    }

    /// `perf.stats.ui` fields: frames counted, the longest, and how many went over 50 / 100 /
    /// 250 ms (cumulative: diff two readings for an interval).
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "frames": self.frames,
            "longestMs": (self.longest_ms * 10.0).round() / 10.0,
            "over50": self.over_50,
            "over100": self.over_100,
            "over250": self.over_250,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_frames_are_counted_by_threshold_and_the_longest_is_kept() {
        let mut t = UiTiming::default();
        for ms in [10.0, 60.0, 120.0, 300.0, 30.0] {
            t.record(ms, ms / 2.0, ms / 2.0);
        }
        let s = t.stats();
        assert_eq!(s["frames"], 5);
        assert_eq!(s["over50"], 3);
        assert_eq!(s["over100"], 2);
        assert_eq!(s["over250"], 1);
        assert_eq!(s["longestMs"], 300.0);
        assert_eq!(t.slow_logged, 1);
    }

    #[test]
    fn a_frame_is_both_halves_and_hostile_values_are_ignored() {
        let mut t = UiTiming::default();
        t.end(Half::Panels); // nothing begun: nothing counted
        assert_eq!(t.stats()["frames"], 0);
        t.begin();
        t.end(Half::Logic);
        t.begin();
        t.end(Half::Panels);
        assert_eq!(t.stats()["frames"], 1);
        t.record(f64::NAN, 0.0, 0.0);
        t.record(-1.0, 0.0, 0.0);
        assert_eq!(t.stats()["frames"], 1);
    }
}
