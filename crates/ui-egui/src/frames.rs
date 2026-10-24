//! Background frame rendering for monitors, thumbnails and playback prefetch.
//!
//! A small pool of worker threads pulls prioritised jobs (the frame on screen first, then the
//! frames playback will need next, then thumbnails). Results land in a byte-budgeted cache keyed
//! by (target, frame, scale, revision); the UI shows the exact frame when ready and otherwise holds
//! the nearest frame it already has, so scrubbing never flashes black and never blocks the UI.
//! Sequence frames inside a rendered segment come from its render preview instead of the live
//! render (see `filmcraft_engine::previews`).

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use web_time::Instant;

use filmcraft_engine::previews::PreviewStore;
use filmcraft_engine::{MediaPool, Services};
use filmcraft_project::{ItemId, Project};
use filmcraft_time::Tick;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// Composite of a sequence at a timeline time.
    Sequence(ItemId),
    /// A single project item at a media time (Source monitor, thumbnails).
    Item(ItemId),
    /// A GPU frame plan of a sequence (decoded layers + transforms), composited on the GPU.
    SequencePlan(ItemId),
    /// The Multi-Camera view's angle grid of a multi-camera source at its time: one page (grid
    /// side, 0 = automatic; page) of the shown angles, each at the job's scale.
    MulticamGrid(ItemId, u8, u16),
    /// One angle of a multi-camera source (Edit Cameras thumbnails).
    MulticamAngle(ItemId, u32),
}

/// Item jobs at this priority or a larger number are background work: thumbnails of the Project
/// panel and the timeline (50) and the Media Browser (40). Monitors and playback use 0..=3. Their
/// decoders are small and are not kept (`filmcraft_media::cancel::with_background`).
pub const BACKGROUND_PRIO: u32 = 40;

/// Whether a job is a thumbnail: a single item at a background priority. Sequence frames are never
/// background, whatever their priority (the scopes ask for the program frame at 40 while playing,
/// and must not turn the program's decoders into small ones).
fn is_background(job: &Job) -> bool {
    background_work(job.prio, job.key.target)
}

fn background_work(prio: u32, target: Target) -> bool {
    prio >= BACKGROUND_PRIO && matches!(target, Target::Item(_))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameKey {
    pub target: Target,
    pub frame: i64,
    /// Output width in pixels (thumbnails) or scale ×1000 (monitors).
    pub size: u32,
    pub revision: u64,
    /// Draft decoding (reduced-resolution playback with Settings ▸ Playback ▸ Draft decoding):
    /// sources may decode approximate non-reference pictures
    /// ([`filmcraft_media::cancel::with_draft`]). Draft frames have their own keys, so a paused
    /// monitor, an export or a render never shows one.
    pub draft: bool,
}

/// Whether a monitor frame is requested with draft decoding ([`FrameKey::draft`]): only while
/// playing, at 1/2 resolution or lower, with the preference on.
pub fn draft_playback(playing: bool, scale: f32, enabled: bool) -> bool {
    enabled && playing && scale <= 0.5
}

pub struct Rgba {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

struct Job {
    key: FrameKey,
    queued: Instant,
    /// Set when the job is no longer wanted; sources poll it (`filmcraft_media::cancel`).
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Queued by playback prefetch (dropped when playback stops).
    prefetch: bool,
    /// While playing: how far before this frame the frames are late (the playhead's distance;
    /// zero for the frame due now), so a decoder catching up may skip pictures only late frames
    /// need (`filmcraft_media::cancel::with_catch_up`).
    catch_up: Option<Tick>,
    time: Tick,
    /// Output scale relative to the target's frame size.
    scale: f32,
    project: Arc<Project>,
    prio: u32,
}

struct Shared {
    queue: Mutex<VecDeque<Job>>,
    cv: Condvar,
    done: Mutex<Cache>,
    plans: Mutex<PlanCache>,
    /// Running jobs: key, cancel flag, prefetch.
    in_flight: Mutex<Vec<(FrameKey, Arc<std::sync::atomic::AtomicBool>, bool)>>,
    /// Per-job timings, collected while profiling is on (benchmarks, `ui.inspect`).
    profiling: AtomicBool,
    records: Mutex<Vec<JobRecord>>,
    /// Running estimate (s) of the per-frame work of playback jobs that is not fetching source
    /// frames: compositing, CPU effects. Frames cannot be ready sooner than this after they are
    /// started, and the workers can finish at most `workers / cost` of them per second.
    render_cost: Mutex<f64>,
    /// What playback jobs learned about throughput-bound sources (see [`SourceRate`]).
    throughput: Mutex<Throughput>,
    workers: usize,
    /// Always-on counters for `perf.stats`.
    stats: Mutex<FrameStats>,
}

/// What playback knows about the throughput-bound sources it is playing (sources whose frames
/// decode independently, [`filmcraft_media::FrameDelivery`]): the frames per second the slowest
/// of them delivers and the time a playback job on one takes from its start to its frame. Drives
/// [`throughput_plan`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SourceRate {
    /// Frames per second (smoothed); `None` until a source reported one or enough jobs completed.
    pub rate: Option<f64>,
    /// Seconds from a job's start to its frame (smoothed), with the wait for a decoder slot.
    pub latency: f64,
}

#[derive(Default)]
struct Throughput {
    model: SourceRate,
    /// A playback job has touched a throughput-bound source, and the ones since did too.
    active: bool,
    /// Playback jobs in a row that touched none: a few of them (another sequence, the clip after
    /// the cut) clear the model, so an ordinary source is not played with a stride.
    misses: u32,
    /// When jobs on such sources finished (not cancelled), for the rate when the source itself
    /// reports none.
    completions: VecDeque<Instant>,
}

impl Throughput {
    /// Playback jobs without a throughput-bound source after which the model is dropped.
    const MISSES: u32 = 12;
    const COMPLETIONS: usize = 32;

    /// A playback job finished: `delivery` is what its sources said (`None`: none of them is
    /// throughput-bound), `decoded` whether it asked any source for a frame at all (a frame past
    /// the end of the sequence, a gap or a rendered segment asks none, and says nothing about
    /// what is playing), `took` its start-to-finish time when it was not cancelled.
    fn job(&mut self, delivery: Option<filmcraft_media::FrameDelivery>, decoded: bool, took: Option<f64>) {
        let Some(d) = delivery.filter(|d| d.independent) else {
            if decoded {
                self.misses = self.misses.saturating_add(1);
                if self.misses >= Self::MISSES {
                    *self = Self::default();
                }
            }
            return;
        };
        self.misses = 0;
        self.active = true;
        let now = Instant::now();
        if let Some(took) = took.filter(|t| t.is_finite() && *t >= 0.0) {
            // late is worse than early: rise fast, settle slowly
            let l = self.model.latency;
            self.model.latency = if l == 0.0 {
                took
            } else if took > l {
                l * 0.5 + took * 0.5
            } else {
                l * 0.8 + took * 0.2
            };
            if self.completions.len() >= Self::COMPLETIONS {
                self.completions.pop_front();
            }
            self.completions.push_back(now);
        }
        let measured = d.rate.filter(|r| r.is_finite() && *r > 0.0).or_else(|| {
            // completions over the last two seconds, once there are enough to mean something
            let recent: Vec<&Instant> = self.completions.iter().filter(|t| now.saturating_duration_since(**t) <= Duration::from_secs(2)).collect();
            let span = recent.first().map(|t| now.saturating_duration_since(**t).as_secs_f64()).unwrap_or(0.0);
            (recent.len() >= 6 && span > 0.0).then(|| (recent.len() - 1) as f64 / span)
        });
        if let Some(r) = measured {
            self.model.rate = Some(match self.model.rate {
                // the source's reading is its best recent moment: believe a rise slowly, a
                // fall quickly, so the stride does not flap on one fast reply
                Some(p) if r > p => p * 0.9 + r * 0.1,
                Some(p) => p * 0.6 + r * 0.4,
                None => r,
            });
        }
    }
}

/// Cumulative frame-job statistics (`perf.stats`): cheap enough to keep on all the time.
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// Jobs finished (including cancelled ones).
    pub jobs: u64,
    /// Jobs whose decode/render panicked (the worker survives; the frame is not cached).
    pub failed: u64,
    pub cancelled: u64,
    /// Jobs served from a render preview.
    pub preview: u64,
    /// Playback prefetch jobs.
    pub prefetch: u64,
    /// Requests answered by a cached frame or one already queued / running.
    pub request_hits: u64,
    /// Requests that queued a new job.
    pub request_misses: u64,
    /// Totals over all jobs (ms): getting source frames (decode), the rest (render), whole job.
    pub source_ms: f64,
    pub render_ms: f64,
    pub job_ms: f64,
    /// The most recent jobs: (source ms, render ms).
    pub recent: VecDeque<(f32, f32)>,
}

impl FrameStats {
    const RECENT: usize = 256;

    /// JSON for `perf.stats`.
    pub fn to_json(&self) -> serde_json::Value {
        let pct = |f: &dyn Fn(&(f32, f32)) -> f32, p: f64| -> f64 {
            let mut v: Vec<f32> = self.recent.iter().map(f).collect();
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(|a, b| a.total_cmp(b));
            v[((v.len() - 1) as f64 * p).round() as usize] as f64
        };
        let n = self.jobs.max(1) as f64;
        let req = self.request_hits + self.request_misses;
        serde_json::json!({
            "jobs": self.jobs, "cancelled": self.cancelled, "failedJobs": self.failed, "previewJobs": self.preview, "prefetchJobs": self.prefetch,
            "requestHitRate": if req == 0 { 0.0 } else { self.request_hits as f64 / req as f64 },
            "decodeMs": {"mean": self.source_ms / n, "p50": pct(&|r| r.0, 0.5), "p95": pct(&|r| r.0, 0.95)},
            "renderMs": {"mean": self.render_ms / n, "p50": pct(&|r| r.1, 0.5), "p95": pct(&|r| r.1, 0.95)},
            "jobMsMean": self.job_ms / n,
        })
    }
}

/// Timing of one finished frame job (collected while [`FrameServer::set_profiling`] is on).
#[derive(Clone, Copy, Debug)]
pub struct JobRecord {
    pub key: FrameKey,
    pub prio: u32,
    pub queued: Instant,
    pub started: Instant,
    pub finished: Instant,
    /// Thread-CPU time of the whole job (robust to machine load; zero where unsupported).
    pub cpu: Duration,
    /// Wall and thread-CPU time spent getting source frames (decode, including waits on a
    /// source's decoder lock; the decoder's own worker threads are not included in `source_cpu`).
    pub source_wall: Duration,
    pub source_cpu: Duration,
    /// The frame came from a render preview.
    pub preview: bool,
    /// The job was cancelled while running (its result was discarded).
    pub cancelled: bool,
}

/// CPU time consumed by the calling thread (None where the platform has no thread clock).
pub fn thread_cpu_time() -> Option<Duration> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        Some(Duration::new(t.tv_sec as u64, t.tv_nsec as u32))
    }
    #[cfg(windows)]
    {
        cpu_time::ThreadTime::try_now().ok().map(|t| t.as_duration())
    }
    #[cfg(not(any(windows, all(unix, not(target_arch = "wasm32")))))]
    {
        None
    }
}

/// CPU time consumed by the whole process, all threads (None where unsupported).
pub fn process_cpu_time() -> Option<Duration> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
        Some(Duration::new(t.tv_sec as u64, t.tv_nsec as u32))
    }
    #[cfg(windows)]
    {
        cpu_time::ProcessTime::try_now().ok().map(|t| t.as_duration())
    }
    #[cfg(not(any(windows, all(unix, not(target_arch = "wasm32")))))]
    {
        None
    }
}

thread_local! {
    /// (wall, cpu) spent in source `video_frame` calls by this worker during the current job.
    static SOURCE_TIME: Cell<(Duration, Duration)> = const { Cell::new((Duration::ZERO, Duration::ZERO)) };
    /// What the throughput-bound sources of the current job said about their delivery (the
    /// slowest rate among them); `None` when none was asked for a frame.
    static DELIVERY: Cell<Option<filmcraft_media::FrameDelivery>> = const { Cell::new(None) };
    /// The current job asked a source for a frame.
    static DECODED: Cell<bool> = const { Cell::new(false) };
    /// This worker is profiling the current job (thread CPU of source calls is measured).
    static PROFILING: Cell<bool> = const { Cell::new(false) };
}

/// Run `f`, adding its wall/CPU time to this thread's source-time accumulator.
fn timed_source<R>(f: impl FnOnce() -> R) -> R {
    let cpu = || if PROFILING.with(Cell::get) { thread_cpu_time().unwrap_or_default() } else { Duration::ZERO };
    let (w0, c0) = (Instant::now(), cpu());
    let r = f();
    let (dw, dc) = (w0.elapsed(), cpu().saturating_sub(c0));
    SOURCE_TIME.with(|s| {
        let (w, c) = s.get();
        s.set((w + dw, c + dc));
    });
    r
}

/// A media source that times its `video_frame` calls (see [`JobRecord::source_wall`]).
struct TimedSource(filmcraft_media::SharedSource);

impl filmcraft_media::MediaSource for TimedSource {
    fn info(&self) -> &filmcraft_media::MediaInfo {
        self.0.info()
    }
    fn video_frame(&self, req: filmcraft_media::FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        let d = self.0.frame_delivery();
        if d.independent {
            DELIVERY.with(|c| {
                let m = c.get().unwrap_or_default();
                let rate = match (m.rate, d.rate) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                c.set(Some(filmcraft_media::FrameDelivery { independent: true, rate }));
            });
        }
        DECODED.with(|c| c.set(true));
        timed_source(|| self.0.video_frame(req))
    }
    fn frame_delivery(&self) -> filmcraft_media::FrameDelivery {
        self.0.frame_delivery()
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.0.audio(start, frames, sample_rate)
    }
    fn audio_stream(&self, stream: usize, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.0.audio_stream(stream, start, frames, sample_rate)
    }
}

/// The job's source provider: the pool, with source fetches timed (the time feeds
/// [`FrameServer::render_cost`] and, while profiling, [`JobRecord`]).
struct JobProvider {
    inner: filmcraft_engine::media_pool::PoolProvider,
}

impl filmcraft_render::SourceProvider for JobProvider {
    fn source(&self, item: ItemId) -> Option<filmcraft_media::SharedSource> {
        let s = self.inner.pool.monitor_source(item, self.inner.source(item)?);
        Some(Arc::new(TimedSource(s)))
    }
}

/// A GPU frame plan with its layers' texel conversions already done (off the UI thread).
pub struct GpuPlan {
    pub plan: filmcraft_render::plan::FramePlan,
    pub prepared: filmcraft_gpu::PreparedPlan,
}

impl GpuPlan {
    /// Bytes this plan keeps alive (layer frames + converted texels).
    fn bytes(&self) -> usize {
        let frames = self.plan.source_bytes();
        frames + self.prepared.bytes()
    }
}

/// Finished plans, bounded by count and bytes (see [`eviction_key`] for which go first).
#[derive(Default)]
struct PlanCache {
    /// Plan, when it was last looked up, when it was inserted, its bytes.
    map: HashMap<FrameKey, (Arc<GpuPlan>, u64, u64, usize)>,
    clock: u64,
    bytes: usize,
    /// Bytes of the plan inserted last: what one frame of what is playing costs.
    last_bytes: usize,
}

/// Frames playback may hold ahead of the playhead when the cache holds `capacity` of them:
/// all of `depth` while they fit, else what fits beside the frame on screen and one behind it.
/// Decoders that run a little faster than the frames are due would otherwise fill the cache
/// with frames far ahead and evict the one due next (a 6K half-resolution plan is 123 MB,
/// eight to the budget, and the prefetch runs 14 deep).
fn lookahead(capacity: usize, depth: i64) -> i64 {
    depth.min(i64::try_from(capacity).unwrap_or(i64::MAX).saturating_sub(2)).max(2)
}

/// Which cache entries go first when the cache is over budget, given when an entry was last
/// looked up and when it was inserted: entries looked up since they were inserted (shown, or
/// scrubbed past), least recently used first; then the ones never looked up, oldest first.
/// Playback prefetches frames ahead of the playhead, and nothing looks at one until it is due:
/// under plain LRU they were the oldest entries, and a 6K half-resolution frame (123 MB as a
/// plan, eight to the budget) decoded 26 frames ahead was evicted before the playhead reached
/// it, while the frames behind the playhead stayed.
fn eviction_key(used: u64, inserted: u64) -> (bool, u64) {
    (used == inserted, used)
}

impl PlanCache {
    const MAX: usize = 96;
    const BUDGET: usize = 1 << 30;

    fn get(&mut self, k: &FrameKey) -> Option<Arc<GpuPlan>> {
        self.clock += 1;
        let clock = self.clock;
        self.map.get_mut(k).map(|(p, used, _, _)| {
            *used = clock;
            p.clone()
        })
    }

    /// How many plans like the last one fit the budget.
    fn capacity(&self) -> usize {
        (Self::BUDGET / self.last_bytes.max(1)).min(Self::MAX)
    }

    fn insert(&mut self, k: FrameKey, p: GpuPlan) {
        self.clock += 1;
        let b = p.bytes();
        self.last_bytes = b;
        self.bytes += b;
        if let Some((_, _, _, old)) = self.map.insert(k, (Arc::new(p), self.clock, self.clock, b)) {
            self.bytes -= old;
        }
        if self.map.len() > Self::MAX || self.bytes > Self::BUDGET {
            let mut v: Vec<((bool, u64), FrameKey, usize)> = self.map.iter().map(|(k, v)| (eviction_key(v.1, v.2), *k, v.3)).collect();
            v.sort_unstable_by_key(|x| x.0);
            for (_, k, b) in v {
                if self.map.len() <= Self::MAX && self.bytes <= Self::BUDGET {
                    break;
                }
                self.map.remove(&k);
                self.bytes -= b;
            }
        }
    }
}

struct Cache {
    map: HashMap<FrameKey, (Arc<Rgba>, u64)>,
    bytes: usize,
    budget: usize,
    clock: u64,
}

impl Cache {
    fn insert(&mut self, k: FrameKey, v: Arc<Rgba>) {
        self.clock += 1;
        self.bytes += v.px.len();
        if let Some((old, _)) = self.map.insert(k, (v, self.clock)) {
            self.bytes -= old.px.len();
        }
        if self.bytes > self.budget {
            let mut all: Vec<(u64, FrameKey, usize)> = self.map.iter().map(|(k, (v, s))| (*s, *k, v.px.len())).collect();
            all.sort_unstable_by_key(|x| x.0);
            for (_, k, b) in all {
                if self.bytes <= self.budget * 8 / 10 {
                    break;
                }
                self.map.remove(&k);
                self.bytes -= b;
            }
        }
    }
}

pub struct FrameServer {
    shared: Arc<Shared>,
    pub pool: Arc<MediaPool>,
    pub services: Arc<dyn Services>,
    pub previews: Arc<PreviewStore>,
    repaint: Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>,
}

/// Frames queued ahead of the playhead while playing (and how far ahead a queued job may be
/// before it is dropped as stale).
pub fn prefetch_depth(speed: f64) -> i64 {
    if speed.abs() > 1.5 { 24 } else { 14 }
}

/// Playback starts its clock once this many frames from the playhead are ready, or after
/// [`PREROLL_TIMEOUT_S`]: otherwise the first frames after Play are due before any decoder had a
/// chance to produce them (a seek decodes from the previous keyframe).
pub const PREROLL_FRAMES: i64 = 6;

/// Seconds before a scrubbed-to frame whose frames a seek still decodes in full (see
/// [`FrameServer::request`]); earlier non-reference pictures are skipped on the way.
pub const SCRUB_KEEP_S: f64 = 2.0;
pub const PREROLL_TIMEOUT_S: f64 = 0.5;

/// Which frames playback asks for, given the per-frame render cost (s), the displayed frame rate
/// and the worker count: `(lead, stride)`. While the workers keep up, every frame from the
/// playhead on (`(1, 1)`). When they cannot (CPU effects slower than real time), `lead` is the
/// first frame ahead of the playhead worth starting (nearer ones would be late however soon they
/// start) and `stride` the spacing the workers can finish in real time: the frames in between
/// are dropped on purpose and evenly, instead of every frame arriving late.
pub fn playback_plan(render_cost: f64, fps: f64, workers: usize) -> (i64, i64) {
    let frames_per_job = render_cost * fps;
    let load = frames_per_job * 1.15 / workers.max(1) as f64;
    if load <= 1.0 {
        return (1, 1);
    }
    (frames_per_job.ceil() as i64, load.ceil() as i64)
}

/// Which frames playback asks for from a throughput-bound source (frames decode independently,
/// [`SourceRate`]): `(lead, stride)` like [`playback_plan`]. `stride` is what the source
/// delivers against what the sequence needs (25 fps under 59.94: every third frame, 20 fps
/// evenly, instead of all of them each a little late), and `lead` its latency in frames with a
/// quarter and a frame of slack, so a frame started that early is ready when it is due.
pub fn throughput_plan(model: SourceRate, fps: f64) -> (i64, i64) {
    let stride = match model.rate {
        // a tenth of a frame of tolerance: a source just short of the rate (23 under 24) is
        // better asked for every frame, dropping the odd late one, than for every other frame
        Some(r) if r > 0.0 && r < fps => (fps / r - 0.1).ceil().max(1.0) as i64,
        _ => 1,
    };
    let lead = (model.latency * fps * 1.25).ceil() as i64 + 1;
    (lead.clamp(1, 600), stride.clamp(1, 64))
}

/// The frames a playback refresh at `cur` asks for: `ahead` of them from `lead` frames on,
/// `stride` apart. `anchored` ones are multiples of the stride, so the set does not shift with
/// the playhead at every refresh: the frames between are left out for good, not asked for a
/// refresh later when they could not be ready either.
fn planned_frames(cur: i64, dir: i64, step: i64, lead: i64, stride: i64, ahead: i64, anchored: bool) -> impl Iterator<Item = i64> {
    let unit = stride.saturating_mul(step).max(1);
    let mut first = cur.saturating_add(lead.saturating_mul(step).saturating_mul(dir));
    if anchored && stride > 1 {
        let r = first.rem_euclid(unit);
        if r != 0 {
            first = first.saturating_add(if dir > 0 { unit - r } else { -r });
        }
    }
    (0..ahead.max(0)).map(move |i| first.saturating_add(i.saturating_mul(unit).saturating_mul(dir)))
}

/// How far behind the playhead a running job on a throughput-bound source is left to finish:
/// its decoder has the frame in hand (cancelling it saves nothing), and the monitor shows a late
/// frame when nothing newer is ready, looking this far back.
pub const LATE_GRACE: i64 = 6;

impl FrameServer {
    /// Settings ▸ Memory ▸ frame cache budget (bytes); evicts down to it right away.
    pub fn set_cache_budget(&self, bytes: usize) {
        let mut c = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
        c.budget = bytes.max(16 << 20);
        if c.bytes > c.budget
            && let Some((k, v)) = c.map.iter().next().map(|(k, (v, _))| (*k, v.clone()))
        {
            // re-inserting an entry runs the eviction pass
            c.insert(k, v);
        }
    }

    /// (bytes cached, budget).
    pub fn cache_usage(&self) -> (usize, usize) {
        let c = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
        (c.bytes, c.budget)
    }

    pub fn new(pool: Arc<MediaPool>, services: Arc<dyn Services>, previews: Arc<PreviewStore>, workers: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            done: Mutex::new(Cache { map: HashMap::new(), bytes: 0, budget: 768 << 20, clock: 0 }),
            plans: Mutex::new(PlanCache::default()),
            in_flight: Mutex::new(Vec::new()),
            profiling: AtomicBool::new(false),
            records: Mutex::new(Vec::new()),
            render_cost: Mutex::new(0.0),
            throughput: Mutex::new(Throughput::default()),
            workers: workers.max(1),
            stats: Mutex::new(FrameStats::default()),
        });
        let repaint: Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>> = Arc::new(Mutex::new(None));
        #[cfg(not(target_arch = "wasm32"))]
        for i in 0..workers.max(1) {
            let sh = shared.clone();
            let pool = pool.clone();
            let services = services.clone();
            let rp = repaint.clone();
            let pv = previews.clone();
            std::thread::Builder::new().name(format!("filmcraft-frames-{i}")).spawn(move || worker(sh, pool, services, pv, rp)).ok();
        }
        #[cfg(target_arch = "wasm32")]
        let _ = workers;
        Self { shared, pool, services, previews, repaint }
    }

    /// The default number of frame workers for this machine.
    pub fn default_workers() -> usize {
        std::thread::available_parallelism().map(|n| n.get().clamp(2, 6)).unwrap_or(3)
    }

    pub fn set_context(&self, ctx: &egui::Context) {
        let mut g = self.repaint.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            let ctx = ctx.clone();
            *g = Some(Box::new(move || ctx.request_repaint()));
        }
    }

    /// Collect a [`JobRecord`] per finished job (benchmarks); `take_records` drains them.
    pub fn set_profiling(&self, on: bool) {
        self.shared.profiling.store(on, Ordering::Relaxed);
    }

    pub fn take_records(&self) -> Vec<JobRecord> {
        std::mem::take(&mut *self.shared.records.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Cumulative job statistics (`perf.stats`).
    pub fn stats(&self) -> FrameStats {
        self.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Worker threads.
    pub fn workers(&self) -> usize {
        self.shared.workers
    }

    /// (cached images, their bytes, cached GPU plans, their bytes).
    pub fn cache_entries(&self) -> (usize, usize, usize, usize) {
        let (n, b) = {
            let c = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
            (c.map.len(), c.bytes)
        };
        let p = self.shared.plans.lock().unwrap_or_else(|e| e.into_inner());
        (n, b, p.map.len(), p.bytes)
    }

    pub fn get(&self, k: &FrameKey) -> Option<Arc<Rgba>> {
        let mut c = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
        c.clock += 1;
        let clock = c.clock;
        c.map.get_mut(k).map(|(v, s)| {
            *s = clock;
            v.clone()
        })
    }

    pub fn get_plan(&self, k: &FrameKey) -> Option<Arc<GpuPlan>> {
        self.shared.plans.lock().unwrap_or_else(|e| e.into_inner()).get(k)
    }

    /// Whether the exact frame (image or plan) is ready.
    pub fn is_ready(&self, k: &FrameKey) -> bool {
        match k.target {
            Target::SequencePlan(_) => self.shared.plans.lock().unwrap_or_else(|e| e.into_inner()).map.contains_key(k),
            _ => self.shared.done.lock().unwrap_or_else(|e| e.into_inner()).map.contains_key(k),
        }
    }

    /// Nearest cached plan at or before `frame`.
    pub fn nearest_plan(&self, key: FrameKey, max_back: i64) -> Option<(FrameKey, Arc<GpuPlan>)> {
        let mut g = self.shared.plans.lock().unwrap_or_else(|e| e.into_inner());
        (0..=max_back).find_map(|d| {
            let k = FrameKey { frame: key.frame - d, ..key };
            g.get(&k).map(|p| (k, p))
        })
    }

    /// Queue a job unless it is cached, queued or in flight.
    ///
    /// The frame on screen (`prio` 0: scrubbing, a seek) treats frames more than
    /// [`SCRUB_KEEP_S`] before it as not wanted: a decoder seeking from a keyframe skips their
    /// non-reference pictures, while nearby frames stay decoded (and cached) for scrubbing back.
    pub fn request(&self, key: FrameKey, time: Tick, scale: f32, project: &Arc<Project>, prio: u32) {
        let late = (prio == 0).then(|| Tick::from_seconds_f64(SCRUB_KEEP_S));
        self.request_job(key, time, scale, project, prio, false, late);
    }

    fn request_job(&self, key: FrameKey, time: Tick, scale: f32, project: &Arc<Project>, prio: u32, prefetch: bool, catch_up: Option<Tick>) {
        let hit = || self.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).request_hits += 1;
        if self.shared.done.lock().unwrap_or_else(|e| e.into_inner()).map.contains_key(&key)
            || self.shared.plans.lock().unwrap_or_else(|e| e.into_inner()).map.contains_key(&key)
        {
            hit();
            return;
        }
        if self.shared.in_flight.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|(k, c, _)| *k == key && !c.load(Ordering::Relaxed)) {
            hit();
            return;
        }
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(j) = q.iter_mut().find(|j| j.key == key) {
            j.prio = j.prio.min(prio);
            j.prefetch &= prefetch;
            // the playhead only moves on: the newest (smallest) margin
            j.catch_up = match (j.catch_up, catch_up) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            drop(q);
            hit();
            return;
        }
        self.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).request_misses += 1;
        if prio == 0 {
            // A new frame on screen for this view replaces the one asked for before: scrubbing
            // asks for one per refresh, and without this the oldest position would decode
            // first (or keep a decoder busy seeking to it) while the newest waits.
            //
            // A job already running for the same frame at an older revision (a value being
            // dragged asks for a new revision per refresh) is left to finish: it needs the same
            // source frames, so it holds no decoder back, and when frames take longer than a
            // refresh, cancelling each one for the next would show nothing until the mouse rests.
            let same_view = |k: &FrameKey| k.target == key.target && k.size == key.size;
            q.retain(|j| !(j.prio == 0 && !j.prefetch && same_view(&j.key)));
            for (k, c, prefetch) in self.shared.in_flight.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                if !prefetch && same_view(k) && k.frame != key.frame {
                    c.store(true, Ordering::Relaxed);
                }
            }
        }
        q.push_back(Job { key, queued: Instant::now(), cancel: Default::default(), prefetch, catch_up, time, scale, project: project.clone(), prio });
        drop(q);
        self.shared.cv.notify_one();
    }

    /// Playback scheduling for the monitor showing `key` (the frame due now): request it first,
    /// then the next frames in display order, and drop queued jobs for this target that are
    /// stale (old revision, behind the playhead or too far ahead).
    ///
    /// When frames cost more to render than the frame interval (CPU effects), frames that could
    /// not be ready in time are not started and the workers get evenly spaced frames they can
    /// finish ([`playback_plan`]). While `preroll` (clock not started yet) every frame is wanted.
    pub fn schedule_playback(&self, key: FrameKey, rate: filmcraft_time::FrameRate, scale: f32, project: &Arc<Project>, speed: f64, preroll: bool) {
        let dir = if speed < 0.0 { -1 } else { 1 };
        let step = speed.abs().max(1.0) as i64;
        let fps = rate.as_f64() * speed.abs().max(1.0) / step as f64;
        let model = self.source_rate();
        // A throughput-bound source (camera RAW) sets the plan when it is the slower constraint.
        let (lead, stride, anchored) = match (preroll, model) {
            (true, _) => (1, 1, false),
            (false, None) => {
                let (l, s) = playback_plan(self.render_cost(), fps, self.shared.workers);
                (l, s, false)
            }
            (false, Some(m)) => {
                let (l, s) = playback_plan(self.render_cost(), fps, self.shared.workers);
                let (lt, st) = throughput_plan(m, fps);
                (l.max(lt), s.max(st), true)
            }
        };
        if lead <= 1 {
            // the frame due now; once the clock runs, the frames before it are late
            self.request_job(key, rate.tick_of(key.frame), scale, project, 0, false, (!preroll && dir > 0).then_some(Tick::ZERO));
        }
        // During preroll the clock has not started: only the frames that gate start, not a
        // prefetch run that walks the rest of a LongGOP and evicts the playhead pictures.
        let depth = lookahead(self.plan_capacity(key.target), prefetch_depth(speed));
        let ahead = if preroll { (PREROLL_FRAMES - 1).max(0) } else { depth };
        let mut prio = 1;
        for f in planned_frames(key.frame, dir, step, lead, stride, ahead, anchored) {
            if f >= 0 {
                // forward play: frames before the playhead are late (reverse play skips nothing)
                let late = (!preroll && dir > 0).then(|| rate.tick_of(f) - rate.tick_of(key.frame));
                self.request_job(FrameKey { frame: f, ..key }, rate.tick_of(f), scale, project, prio, true, late);
            }
            prio += 1;
        }
        let mut span = (lead + ahead * stride) * step + 26;
        if preroll && let Some(m) = model {
            // A throughput-bound source's planned run as well, from right after the preroll
            // frames: the clock has not started, so every frame decoded now is ahead of it,
            // and the clock must not start into frames nobody asked for.
            let (lt, st) = throughput_plan(m, fps);
            let (lt, n) = (lt.min(PREROLL_FRAMES), depth);
            for f in planned_frames(key.frame, dir, step, lt, st, n, true) {
                if f >= 0 {
                    self.request_job(FrameKey { frame: f, ..key }, rate.tick_of(f), scale, project, prio, true, None);
                }
                prio += 1;
            }
            span = span.max((lt + n * st) * step + 26);
        }
        let (target, rev, cur) = (key.target, key.revision, key.frame);
        // Queued frames behind the playhead are late; a running one on a throughput-bound
        // source is left to finish while it is within the grace (its frame is still shown).
        let grace = if model.is_some() { LATE_GRACE } else { 0 };
        let wanted = |k: &FrameKey, behind: i64| k.target != target || (k.revision == rev && (k.frame - cur) * dir >= -behind && (k.frame - cur).abs() < span);
        self.retain_playback(|k| wanted(k, 0), |k| wanted(k, grace));
    }

    /// Whether playback can start at `key`: the next [`PREROLL_FRAMES`] frames (up to `last`)
    /// are ready.
    pub fn preroll_ready(&self, key: FrameKey, speed: f64, last: i64) -> bool {
        let step = (speed.abs().max(1.0) as i64) * if speed < 0.0 { -1 } else { 1 };
        (0..PREROLL_FRAMES).map(|i| key.frame + i * step).filter(|f| *f >= 0 && *f <= last).all(|f| self.is_ready(&FrameKey { frame: f, ..key }))
    }

    /// The current per-frame render cost estimate (s), see [`playback_plan`].
    pub fn render_cost(&self) -> f64 {
        *self.shared.render_cost.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// How many frames of `target` its cache holds (see [`lookahead`]); only GPU plans are
    /// large enough to matter, the image cache's entries are 8-bit at the monitor's size.
    fn plan_capacity(&self, target: Target) -> usize {
        match target {
            Target::SequencePlan(_) => self.shared.plans.lock().unwrap_or_else(|e| e.into_inner()).capacity(),
            _ => usize::MAX,
        }
    }

    /// What playback learned about the throughput-bound sources it is playing, while it is
    /// playing one (see [`throughput_plan`]); `None` for ordinary sources.
    pub fn source_rate(&self) -> Option<SourceRate> {
        let t = self.shared.throughput.lock().unwrap_or_else(|e| e.into_inner());
        t.active.then_some(t.model)
    }

    /// The plan playback at `fps` would use now: `(lead, stride)` (see [`playback_plan`] and
    /// [`throughput_plan`]), for `perf.stats`.
    pub fn plan_for(&self, fps: f64) -> (i64, i64) {
        let (l, s) = playback_plan(self.render_cost(), fps, self.shared.workers);
        match self.source_rate() {
            Some(m) => {
                let (lt, st) = throughput_plan(m, fps);
                (l.max(lt), s.max(st))
            }
            None => (l, s),
        }
    }

    /// Drop queued jobs that fail `keep` (e.g. stale revisions or frames far from the playhead).
    /// Jobs already running for such frames are cancelled: a worker blocked behind a decoder
    /// would otherwise go on to decode a frame playback has passed, possibly from its keyframe.
    pub fn retain_queue(&self, keep: impl Fn(&FrameKey) -> bool) {
        self.retain_playback(&keep, &keep);
    }

    /// [`retain_queue`](Self::retain_queue) with its own rule for running jobs.
    fn retain_playback(&self, keep_queued: impl Fn(&FrameKey) -> bool, keep_running: impl Fn(&FrameKey) -> bool) {
        self.shared.queue.lock().unwrap_or_else(|e| e.into_inner()).retain(|j| keep_queued(&j.key));
        for (k, c, _) in self.shared.in_flight.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if !keep_running(k) {
                c.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Playback stopped: drop its queued prefetch jobs and cancel the running ones (left alone,
    /// they would keep the workers busy, e.g. rendering effects for frames nobody will see).
    pub fn stop_prefetch(&self) {
        self.shared.queue.lock().unwrap_or_else(|e| e.into_inner()).retain(|j| !j.prefetch);
        for (_, c, prefetch) in self.shared.in_flight.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if *prefetch {
                c.store(true, Ordering::Relaxed);
            }
        }
    }

    pub fn queue_len(&self) -> usize {
        self.shared.queue.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// The most recent cached frame for `target` at or before `frame` within `max_back` frames.
    pub fn nearest(&self, target: Target, frame: i64, size: u32, revision: u64, max_back: i64) -> Option<Arc<Rgba>> {
        for d in 0..=max_back {
            if let Some(v) = self.get(&FrameKey { target, frame: frame - d, size, revision, draft: false }) {
                return Some(v);
            }
        }
        None
    }

    /// Without worker threads (wasm32): run queued jobs on the calling (UI) thread, highest
    /// priority first, until `budget` has passed (at least one job). Jobs whose media is still
    /// loading go back to the queue. Returns the number of jobs finished. A no-op with workers.
    pub fn pump(&self, budget: Duration) -> usize {
        if cfg!(not(target_arch = "wasm32")) {
            return 0;
        }
        let t0 = Instant::now();
        let mut done = 0;
        let mut retry = Vec::new();
        loop {
            let job = {
                let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
                if q.is_empty() {
                    break;
                }
                let best = q.iter().enumerate().min_by_key(|(i, j)| (j.prio, *i)).map(|(i, _)| i).unwrap_or(0);
                match q.remove(best) {
                    Some(job) => job,
                    None => break,
                }
            };
            if run_job(&self.shared, &job, &self.pool, &self.services, &self.previews) {
                done += 1;
            } else {
                retry.push(job);
            }
            if t0.elapsed() >= budget {
                break;
            }
        }
        if !retry.is_empty() {
            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            for j in retry.into_iter().rev() {
                if !q.iter().any(|x| x.key == j.key) {
                    q.push_front(j);
                }
            }
        }
        done
    }
}

/// Counts shown and dropped frames while playing.
///
/// A frame of the timeline is *shown* when its exact picture was on screen at some refresh while
/// it was the frame due, and *dropped* otherwise — including frames the playhead skipped because
/// no refresh happened during their interval. Frames left out on purpose (fast-forward speeds
/// step over frames) are not counted.
#[derive(Clone, Debug, Default)]
pub struct PlaybackMeter {
    pub shown: u64,
    pub dropped: u64,
    /// Frame due at the last refresh, and whether its exact picture has been shown.
    current: Option<(i64, bool)>,
    /// Frames advanced per displayed frame (speed), and direction.
    step: i64,
}

impl PlaybackMeter {
    pub fn start(&mut self, speed: f64) {
        *self = Self { step: (speed.abs().max(1.0) as i64) * if speed < 0.0 { -1 } else { 1 }, ..Default::default() };
    }

    /// Record one display refresh: `frame` is due now and `exact` says whether it is on screen.
    pub fn refresh(&mut self, frame: i64, exact: bool) {
        let step = if self.step == 0 { 1 } else { self.step };
        match self.current {
            Some((f, seen)) if f == frame => self.current = Some((f, seen || exact)),
            Some((f, seen)) => {
                self.close(seen);
                let moved = (frame - f) / step;
                // Forward progress: count frames passed over without a refresh. A jump backwards
                // (loop restart) or a huge jump is a discontinuity, not a drop.
                if moved > 1 && moved < 1000 {
                    self.dropped += (moved - 1) as u64;
                }
                self.current = Some((frame, exact));
            }
            None => self.current = Some((frame, exact)),
        }
    }

    fn close(&mut self, seen: bool) {
        if seen {
            self.shown += 1;
        } else {
            self.dropped += 1;
        }
    }

    /// Continue from `frame` without counting what happened since the last refresh (the window
    /// was hidden, so nothing could be shown).
    pub fn resync(&mut self, frame: i64, exact: bool) {
        self.current = Some((frame, exact));
    }

    /// Playback stopped: account for the frame on screen.
    pub fn finish(&mut self) {
        if let Some((_, seen)) = self.current.take() {
            self.close(seen);
        }
    }

    /// Shown and dropped so far, counting the frame currently due as shown when it is.
    pub fn counts(&self) -> (u64, u64) {
        match self.current {
            Some((_, true)) => (self.shown + 1, self.dropped),
            _ => (self.shown, self.dropped),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn worker(
    sh: Arc<Shared>,
    pool: Arc<MediaPool>,
    services: Arc<dyn Services>,
    previews: Arc<PreviewStore>,
    repaint: Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>,
) {
    loop {
        let job = {
            let mut q = sh.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if !q.is_empty() {
                    // highest priority (lowest number) first; FIFO among equals
                    let best = q.iter().enumerate().min_by_key(|(i, j)| (j.prio, *i)).map(|(i, _)| i).unwrap_or(0);
                    if let Some(job) = q.remove(best) {
                        break job;
                    }
                }
                q = sh.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        // A panic while decoding or rendering one frame (bad media, a decoder bug) must not take
        // the worker down with it: a dead worker leaves its frame "in flight" forever and the
        // monitors blank. Log it, forget the job and keep serving.
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(&sh, &job, &pool, &services, &previews)));
        if ran.is_err() {
            sh.in_flight.lock().unwrap_or_else(|e| e.into_inner()).retain(|(k, c, _)| !(*k == job.key && Arc::ptr_eq(c, &job.cancel)));
            sh.stats.lock().unwrap_or_else(|e| e.into_inner()).failed += 1;
            log::error!("frame job {:?} panicked; the worker continues", job.key);
        }
        if let Some(f) = repaint.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            f();
        }
    }
}

/// Run one job and cache its result. Returns false when the job's media was still loading (an
/// asynchronous web read): nothing was cached and the job should be retried.
/// Fault injection for robustness tests (and agents checking recovery): the next `n` frame jobs
/// panic as if a decoder had hit a bug.
#[doc(hidden)]
pub fn inject_job_panics(n: u32) {
    INJECT_PANICS.store(n, Ordering::Relaxed);
}
static INJECT_PANICS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn run_job(sh: &Shared, job: &Job, pool: &Arc<MediaPool>, services: &Arc<dyn Services>, previews: &PreviewStore) -> bool {
    sh.in_flight.lock().unwrap_or_else(|e| e.into_inner()).push((job.key, job.cancel.clone(), job.prefetch));
    if INJECT_PANICS.load(Ordering::Relaxed) > 0 && INJECT_PANICS.fetch_sub(1, Ordering::Relaxed) > 0 {
        crate::crash::injected_fault("injected frame-job fault");
    }
    let profiling = sh.profiling.load(Ordering::Relaxed);
    let (started, cpu0) = (Instant::now(), if profiling { thread_cpu_time().unwrap_or_default() } else { Duration::ZERO });
    SOURCE_TIME.with(|s| s.set((Duration::ZERO, Duration::ZERO)));
    DELIVERY.with(|d| d.set(None));
    DECODED.with(|d| d.set(false));
    PROFILING.with(|p| p.set(profiling));
    // A cancelled job may have missed layers (its source gave up): its result is not cached.
    // A job whose media bytes were still loading (web) rendered with missing layers: like a
    // cancelled one, its result is not cached (the caller retries it).
    let _ = filmcraft_media::pending::take();
    let mut loading = false;
    let preview = filmcraft_media::cancel::with_cancel(&job.cancel, || {
        filmcraft_media::cancel::with_catch_up(job.catch_up, || {
            filmcraft_media::cancel::with_draft(job.key.draft, || {
                filmcraft_media::cancel::with_background(is_background(job), || {
                    if let Target::SequencePlan(seq) = job.key.target {
                        let (plan, pv) = plan_job(job, seq, pool, services, previews);
                        loading = filmcraft_media::pending::take();
                        if !job.cancel.load(Ordering::Relaxed) && !loading {
                            // Convert texels for upload here, not on the UI thread when the frame is shown.
                            let prepared = filmcraft_gpu::prepare(&plan);
                            sh.plans.lock().unwrap_or_else(|e| e.into_inner()).insert(job.key, GpuPlan { plan, prepared });
                        }
                        pv
                    } else {
                        let (img, pv) = render_job(job, pool, services, previews);
                        loading = filmcraft_media::pending::take();
                        if !job.cancel.load(Ordering::Relaxed) && !loading {
                            sh.done.lock().unwrap_or_else(|e| e.into_inner()).insert(job.key, Arc::new(img));
                        }
                        pv
                    }
                })
            })
        })
    });
    let cancelled = job.cancel.load(Ordering::Relaxed);
    sh.in_flight.lock().unwrap_or_else(|e| e.into_inner()).retain(|(k, c, _)| !(*k == job.key && Arc::ptr_eq(c, &job.cancel)));
    if job.prefetch {
        // Work beyond fetching source frames (decoding is sequential per source and is
        // not saved by skipping frames, so it is left out of the estimate). A cancelled job
        // only tells that the work takes at least this long.
        //
        // Sources whose frames decode independently are the exception, and a per-job cost is
        // the wrong model for them too (tried on a 6K camera RAW source under a 59.94 fps
        // sequence: the job time includes queueing behind the other workers and the plan's lead
        // had no slack, so every planned frame arrived late). They get a rate and a latency of
        // their own ([`SourceRate`], [`throughput_plan`]).
        let cost = started.elapsed().saturating_sub(SOURCE_TIME.with(|s| s.get()).0).as_secs_f64();
        let mut c = sh.render_cost.lock().unwrap_or_else(|e| e.into_inner());
        if !cancelled || cost > *c {
            *c = if *c == 0.0 { cost } else { *c * 0.8 + cost * 0.2 };
        }
        drop(c);
        let took = (!cancelled).then(|| started.elapsed().as_secs_f64());
        sh.throughput.lock().unwrap_or_else(|e| e.into_inner()).job(DELIVERY.with(Cell::take), DECODED.with(Cell::get), took);
    }
    {
        let job_wall = started.elapsed();
        let source = SOURCE_TIME.with(|s| s.get()).0;
        let (src_ms, render_ms) = (source.as_secs_f64() * 1e3, job_wall.saturating_sub(source).as_secs_f64() * 1e3);
        let mut st = sh.stats.lock().unwrap_or_else(|e| e.into_inner());
        st.jobs += 1;
        st.cancelled += cancelled as u64;
        st.preview += preview as u64;
        st.prefetch += job.prefetch as u64;
        st.source_ms += src_ms;
        st.render_ms += render_ms;
        st.job_ms += job_wall.as_secs_f64() * 1e3;
        if st.recent.len() >= FrameStats::RECENT {
            st.recent.pop_front();
        }
        st.recent.push_back((src_ms as f32, render_ms as f32));
    }
    if profiling {
        let (source_wall, source_cpu) = SOURCE_TIME.with(|s| s.get());
        let rec = JobRecord {
            key: job.key,
            prio: job.prio,
            queued: job.queued,
            started,
            finished: Instant::now(),
            cpu: thread_cpu_time().unwrap_or_default().saturating_sub(cpu0),
            source_wall,
            source_cpu,
            preview,
            cancelled,
        };
        sh.records.lock().unwrap_or_else(|e| e.into_inner()).push(rec);
    }
    !loading
}

/// The preview frame for a sequence job, when its segment has been rendered.
fn preview_frame(job: &Job, seq: ItemId, pool: &MediaPool, previews: &PreviewStore) -> Option<Arc<filmcraft_frame::VideoFrame>> {
    let rate = job.project.sequence(seq)?.settings.frame_rate;
    timed_source(|| previews.frame(pool, &job.project, seq, rate.frame_at(job.time), job.scale))
}

fn provider(job: &Job, pool: &Arc<MediaPool>, services: &Arc<dyn Services>) -> JobProvider {
    JobProvider { inner: pool.provider(job.project.clone(), services.clone()) }
}

fn plan_job(job: &Job, seq: ItemId, pool: &Arc<MediaPool>, services: &Arc<dyn Services>, previews: &PreviewStore) -> (filmcraft_render::plan::FramePlan, bool) {
    let opts = filmcraft_render::RenderOptions { scale: job.scale, captions: true, ..Default::default() };
    if let Some(frame) = preview_frame(job, seq, pool, previews)
        && let Some(q) = job.project.sequence(seq)
    {
        // One full-frame layer: the preview scaled to the output size. Previews never contain
        // captions (they are not part of the preview hash), so captions are layered on live.
        let (w, h) = filmcraft_render::output_size(q, job.scale);
        let matrix = filmcraft_geom::Affine::scale(w as f64 / frame.width.max(1) as f64, h as f64 / frame.height.max(1) as f64);
        let mut layers = vec![filmcraft_render::plan::PlanLayer { frame, matrix, opacity: 1.0, blend: filmcraft_render::Blend::Normal, fx: None, input: None }];
        for o in filmcraft_render::caption_overlays(q, job.time, w, h) {
            layers.push(filmcraft_render::plan::PlanLayer {
                frame: Arc::new(filmcraft_frame::VideoFrame::rgba_f32(o.w as u32, o.h as u32, o.px)),
                matrix: filmcraft_geom::Affine::translate(o.x as f64, o.y as f64),
                opacity: 1.0,
                blend: filmcraft_render::Blend::Normal,
                fx: None,
                input: None,
            });
        }
        return (filmcraft_render::plan::FramePlan::Layers { width: w, height: h, layers }, true);
    }
    let provider = provider(job, pool, services);
    (filmcraft_render::plan::plan_frame(&job.project, seq, job.time, opts, &provider), false)
}

/// Render a job to RGBA8; the flag says whether it came from a render preview.
fn render_job(job: &Job, pool: &Arc<MediaPool>, services: &Arc<dyn Services>, previews: &PreviewStore) -> (Rgba, bool) {
    if let Target::Sequence(seq) | Target::SequencePlan(seq) = job.key.target
        && let Some(f) = preview_frame(job, seq, pool, previews)
        && let Some(q) = job.project.sequence(seq)
    {
        let (w, h) = filmcraft_render::output_size(q, job.scale);
        let img = filmcraft_render::Image { w: f.width as usize, h: f.height as usize, px: f.to_linear_f32() };
        let mut img = if img.w == w && img.h == h {
            img
        } else {
            img.transformed(w, h, &filmcraft_geom::Affine::scale(w as f64 / img.w.max(1) as f64, h as f64 / img.h.max(1) as f64))
        };
        if matches!(job.key.target, Target::Sequence(_)) {
            for o in filmcraft_render::caption_overlays(q, job.time, w, h) {
                o.composite_onto(&mut img.px, w, h);
            }
        }
        return (Rgba { w: img.w, h: img.h, px: img.over_black_rgba8() }, true);
    }
    let provider = provider(job, pool, services);
    let opts = filmcraft_render::RenderOptions { scale: job.scale, captions: true, ..Default::default() };
    let img = match job.key.target {
        Target::Sequence(seq) | Target::SequencePlan(seq) => Some(filmcraft_render::render_sequence(&job.project, seq, job.time, opts, &provider)),
        Target::Item(item) => filmcraft_render::render_item(&job.project, item, job.time, job.scale, &provider),
        Target::MulticamGrid(item, side, page) => {
            let side = (side > 0).then_some(side as usize);
            filmcraft_render::multicam::render_grid_page(&job.project, item, job.time, job.scale, side, page as usize, &provider).map(|(img, _)| img)
        }
        Target::MulticamAngle(item, angle) => {
            filmcraft_render::multicam::render_angle_thumbnail(&job.project, item, angle as usize, job.time, job.scale, &provider)
        }
    };
    let rgba = match img {
        Some(img) => Rgba { w: img.w, h: img.h, px: img.over_black_rgba8() },
        None => Rgba { w: 1, h: 1, px: vec![0, 0, 0, 255] },
    };
    (rgba, false)
}

#[cfg(test)]
mod tests {
    use super::{PlaybackMeter, SourceRate, Target, Throughput, background_work, eviction_key, lookahead, planned_frames, playback_plan, throughput_plan};
    use filmcraft_media::FrameDelivery;
    use filmcraft_project::ItemId;

    #[test]
    fn only_item_thumbnails_are_background_work() {
        let item = Target::Item(ItemId(1));
        // thumbnails (50) and the Media Browser (40)
        assert!(background_work(50, item));
        assert!(background_work(40, item));
        // the Source monitor shows an item at a foreground priority
        assert!(!background_work(1, item));
        // the scopes ask for the program frame at 40 while playing: never background
        assert!(!background_work(40, Target::Sequence(ItemId(1))));
        assert!(!background_work(50, Target::SequencePlan(ItemId(1))));
    }

    #[test]
    fn draft_decoding_only_for_reduced_resolution_playback() {
        use super::draft_playback;
        assert!(draft_playback(true, 0.5, true) && draft_playback(true, 0.25, true));
        assert!(!draft_playback(true, 1.0, true), "full resolution is never draft");
        assert!(!draft_playback(false, 0.5, true), "a paused frame is never draft");
        assert!(!draft_playback(true, 0.5, false), "off unless enabled");
        assert!(!filmcraft_engine::autosave::Preferences::default().playback.draft_decode, "off by default");
    }

    #[test]
    fn playback_plan_skips_frames_only_when_workers_cannot_keep_up() {
        // decode-only / light frames: every frame from the playhead on
        assert_eq!(playback_plan(0.0, 24.0, 6), (1, 1));
        assert_eq!(playback_plan(0.15, 24.0, 6), (1, 1));
        // 0.5 s of effects per frame at 24 fps on 6 workers: 12 frames pass while one renders,
        // so start 12 ahead, and every 3rd frame is what 6 workers can finish
        assert_eq!(playback_plan(0.5, 24.0, 6), (12, 3));
    }

    #[test]
    fn throughput_plan_asks_for_what_the_source_delivers_early_enough() {
        // a 25 fps camera RAW decoder under 59.94 fps, 240 ms from a job's start to its frame:
        // every third frame (20 fps, evenly), started 19 frames ahead (240 ms × 1.25 + 1 frame)
        let (lead, stride) = throughput_plan(SourceRate { rate: Some(25.0), latency: 0.24 }, 59.94);
        assert_eq!((lead, stride), (19, 3));
        // keeping up: every frame, still started early enough
        assert_eq!(throughput_plan(SourceRate { rate: Some(30.0), latency: 0.1 }, 29.97), (5, 1));
        // just short of the rate: every frame, the odd late one is dropped (cheaper than half)
        assert_eq!(throughput_plan(SourceRate { rate: Some(23.0), latency: 0.1 }, 24.0).1, 1);
        assert_eq!(throughput_plan(SourceRate { rate: Some(31.0), latency: 0.1 }, 60.0).1, 2);
        assert_eq!(throughput_plan(SourceRate { rate: Some(20.0), latency: 0.1 }, 60.0).1, 3);
        // nothing measured yet: every frame from the playhead on
        assert_eq!(throughput_plan(SourceRate::default(), 59.94), (1, 1));
        assert_eq!(throughput_plan(SourceRate { rate: None, latency: 0.24 }, 59.94).1, 1);
        // hostile values stay bounded
        assert_eq!(throughput_plan(SourceRate { rate: Some(0.001), latency: 1e9 }, 60.0), (600, 64));
        assert_eq!(throughput_plan(SourceRate { rate: Some(-1.0), latency: f64::NAN }, 60.0).1, 1);
    }

    #[test]
    fn anchored_plan_keeps_the_same_frames_as_the_playhead_moves() {
        let frames = |cur, dir, anchored| planned_frames(cur, dir, 1, 19, 3, 4, anchored).collect::<Vec<_>>();
        // multiples of the stride: a refresh one frame later asks for the same frames
        assert_eq!(frames(100, 1, true), vec![120, 123, 126, 129]);
        assert_eq!(frames(101, 1, true), vec![120, 123, 126, 129]);
        assert_eq!(frames(102, 1, true), vec![123, 126, 129, 132]);
        // the plain plan shifts with every refresh (every frame gets asked for eventually)
        assert_eq!(frames(100, 1, false), vec![119, 122, 125, 128]);
        assert_eq!(frames(101, 1, false), vec![120, 123, 126, 129]);
        // reverse play anchors the same way
        assert_eq!(frames(100, -1, true), vec![81, 78, 75, 72]);
        assert_eq!(frames(99, -1, true), vec![78, 75, 72, 69]);
        // 2× speed: the stride is in played frames
        assert_eq!(planned_frames(100, 1, 2, 19, 3, 3, true).collect::<Vec<_>>(), vec![138, 144, 150]);
        assert_eq!(planned_frames(0, 1, 1, 1, 1, 3, true).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(planned_frames(i64::MAX, 1, 1, 1, 1, 2, true).count(), 2, "no overflow");
    }

    #[test]
    fn throughput_model_follows_the_source_and_clears_for_ordinary_ones() {
        let raw = Some(FrameDelivery { independent: true, rate: Some(25.0) });
        let mut t = Throughput::default();
        assert!(!t.active);
        t.job(raw, true, Some(0.2));
        assert!(t.active);
        assert_eq!(t.model, SourceRate { rate: Some(25.0), latency: 0.2 });
        // latency rises fast and settles slowly; a cancelled job (no time) still keeps it active
        t.job(raw, true, Some(0.4));
        assert!((t.model.latency - 0.3).abs() < 1e-9);
        t.job(raw, true, None);
        assert!(t.active);
        // a source without a rate of its own: measured from completions once there are enough
        let mut u = Throughput::default();
        for _ in 0..5 {
            u.job(Some(FrameDelivery { independent: true, rate: None }), true, Some(0.05));
        }
        assert_eq!(u.model.rate, None);
        u.job(Some(FrameDelivery { independent: true, rate: None }), true, Some(0.05));
        assert!(u.model.rate.is_some_and(|r| r > 100.0), "{:?}", u.model);
        // frames that decode nothing (past the end of the sequence while it loops, a gap, a
        // rendered segment) say nothing about what is playing: the model stays
        for _ in 0..(4 * Throughput::MISSES) {
            t.job(None, false, Some(0.001));
        }
        assert!(t.active);
        // playback moves on to a sequence of ordinary sources: a dozen jobs clear the model
        for i in 0..Throughput::MISSES {
            assert!(t.active, "{i}");
            t.job(None, true, Some(0.01));
        }
        assert!(!t.active);
        assert_eq!(t.model, SourceRate::default());
        // a GOP source's delivery is not throughput-bound either
        t.job(Some(FrameDelivery::default()), true, Some(0.01));
        assert!(!t.active);
    }

    #[test]
    fn lookahead_fits_the_plan_cache() {
        // 1080p plans by the hundred: the prefetch depth as it is
        assert_eq!(lookahead(128, 14), 14);
        assert_eq!(lookahead(usize::MAX, 24), 24);
        // eight 6K half-resolution plans: six ahead, the frame on screen and one behind
        assert_eq!(lookahead(8, 14), 6);
        // never less than two, whatever the plan costs
        assert_eq!(lookahead(1, 14), 2);
        assert_eq!(lookahead(0, 14), 2);
    }

    #[test]
    fn prefetched_frames_outlive_the_frames_already_shown() {
        // (name, last looked up, inserted): the playhead's frame is looked up every refresh,
        // the frames behind it were, the frames ahead never
        let mut entries = [("ahead-near", 5, 5), ("ahead-far", 8, 8), ("shown-now", 9, 4), ("shown-before", 7, 3), ("shown-long-ago", 2, 1)];
        entries.sort_by_key(|e| eviction_key(e.1, e.2));
        let order: Vec<&str> = entries.iter().map(|e| e.0).collect();
        assert_eq!(order, ["shown-long-ago", "shown-before", "shown-now", "ahead-near", "ahead-far"]);
    }

    #[test]
    fn meter_resync_does_not_count_hidden_time() {
        let mut m = PlaybackMeter::default();
        m.start(1.0);
        m.refresh(0, true);
        m.resync(60, false); // window hidden for 2.5 s
        m.refresh(60, true);
        m.refresh(61, true);
        m.finish();
        assert_eq!((m.shown, m.dropped), (2, 0));
    }

    #[test]
    fn meter_counts_frames_not_refreshes() {
        // 24 fps on a 60 Hz display: each frame is due for 2–3 refreshes.
        let mut m = PlaybackMeter::default();
        m.start(1.0);
        for (f, exact) in [(0, true), (0, true), (0, true), (1, false), (1, true), (2, false), (2, false), (3, true)] {
            m.refresh(f, exact);
        }
        m.finish();
        // frame 1 arrived late but was on screen while due → shown; frame 2 never → dropped.
        assert_eq!((m.shown, m.dropped), (3, 1));
    }

    #[test]
    fn meter_counts_skipped_frames_but_not_speed_steps_or_loops() {
        let mut m = PlaybackMeter::default();
        m.start(1.0);
        m.refresh(10, true);
        m.refresh(14, true); // no refresh during 11..13 (UI stalled)
        m.refresh(0, true); // loop restart
        m.finish();
        assert_eq!((m.shown, m.dropped), (3, 3));
        let mut m = PlaybackMeter::default();
        m.start(2.0); // 2× steps over every other frame on purpose
        for f in [0, 2, 4, 6] {
            m.refresh(f, true);
        }
        m.finish();
        assert_eq!((m.shown, m.dropped), (4, 0));
    }
}
