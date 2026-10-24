//! `file.import {async: true}`: the media files of an import are opened on a background thread
//! and added to the project, in the order given, as they finish ([`poll`], once per UI frame).
//!
//! Opening is the slow part of an import: some decoders take ~10 ms a clip and open one clip at a
//! time, so a 109-clip card is over a second. That second used to pass on the UI thread, which runs
//! every command the user gives: the window froze for each import. The interactive entry points (the Import dialog, the
//! Media Browser, files dropped on the window) ask for this; scripted callers keep the synchronous
//! import, whose result lists every item.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use filmcraft_project::{BinId, ItemId};
use serde_json::{Value, json};

use crate::commands::{Prefetched, import_prefetched, open_concurrently};
use crate::{EngineError, Result, Session};

/// An import whose files are still being opened.
pub struct Pending {
    pub job: u64,
    bin: Option<BinId>,
    paths: Vec<String>,
    /// The next path to add: results are added in the order of `paths`.
    next: usize,
    opened: Arc<Mutex<HashMap<String, Prefetched>>>,
    items: Vec<ItemId>,
    errors: Vec<String>,
    started: web_time::Instant,
}

/// Start opening `paths` on a background thread for an import into `bin`; returns the job id
/// (its progress shows in the Progress panel).
pub(crate) fn start(s: &mut Session, paths: Vec<String>, bin: Option<BinId>) -> Result<u64> {
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let n = paths.len();
    let job = crate::Job {
        id,
        label: if n == 1 { "Import 1 file".into() } else { format!("Import {n} files") },
        progress: Default::default(),
        result: Default::default(),
    };
    job.progress.total.store(n as u64, std::sync::atomic::Ordering::Relaxed);
    let opened: Arc<Mutex<HashMap<String, Prefetched>>> = Arc::default();
    let (progress, result, sink) = (job.progress.clone(), job.result.clone(), opened.clone());
    let (media, services, todo) = (s.media.clone(), s.services.clone(), paths.clone());
    let run = move || {
        let t0 = web_time::Instant::now();
        open_concurrently(&media, &*services, &todo, &|path: &str, r: Prefetched| {
            sink.lock().unwrap_or_else(|e| e.into_inner()).insert(path.to_string(), r);
            progress.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *progress.status.lock().unwrap_or_else(|e| e.into_inner()) =
                std::path::Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        });
        let secs = t0.elapsed().as_secs_f64();
        *progress.status.lock().unwrap_or_else(|e| e.into_inner()) = format!("Opened in {secs:.1}s");
        progress.finished.store(true, std::sync::atomic::Ordering::Relaxed);
        *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ok(filmcraft_export::Report {
            path: String::new(),
            frames: todo.len() as u64,
            seconds: secs,
            bytes: 0,
            render_fps: 0.0,
            extra_files: Vec::new(),
        }));
    };
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    s.jobs.push(job);
    s.background_imports.push(Pending { job: id, bin, paths, next: 0, opened, items: Vec::new(), errors: Vec::new(), started: web_time::Instant::now() });
    std::thread::Builder::new().name("filmcraft-import".into()).spawn(run).map_err(|e| EngineError::Other(format!("import thread: {e}")))?;
    Ok(id)
}

/// Add what the background imports have opened, in order, and finish those that are done (call
/// regularly, e.g. once per UI frame).
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.background_imports.len() {
        // take the results that are next in order, then add them without the lock
        let ready: Vec<(String, Prefetched)> = {
            let pending = &mut s.background_imports[i];
            let mut opened = pending.opened.lock().unwrap_or_else(|e| e.into_inner());
            let mut ready = Vec::new();
            while let Some(path) = pending.paths.get(pending.next) {
                let Some(r) = opened.remove(path) else { break };
                ready.push((path.clone(), r));
                pending.next += 1;
            }
            ready
        };
        let bin = s.background_imports[i].bin;
        for (path, r) in ready {
            match import_prefetched(s, &path, r, bin) {
                Ok(id) => s.background_imports[i].items.push(id),
                Err(e) => s.background_imports[i].errors.push(format!("{path}: {e}")),
            }
        }
        if s.background_imports[i].next >= s.background_imports[i].paths.len() {
            let done = s.background_imports.remove(i);
            finish(s, done);
        } else {
            i += 1;
        }
    }
}

/// An import whose files are all added: ingest and transcription as the synchronous import does,
/// relink what came back, and say how it went.
fn finish(s: &mut Session, done: Pending) {
    let mut errors = done.errors;
    errors.extend(crate::commands::after_import(s, &done.items));
    s.events.push(crate::Event::ImportFinished { job: done.job, items: done.items.clone() });
    let secs = done.started.elapsed().as_secs_f64();
    log::info!("import of {} files done in {secs:.1} s: {} added, {} failed", done.paths.len(), done.items.len(), errors.len());
    if !errors.is_empty() {
        s.error_toast("file.import", errors.join("; "));
    }
}

/// What `file.import` reports about a background import.
pub(crate) fn describe(job: u64, pending: usize) -> Value {
    json!({ "importJob": job, "pending": pending })
}
