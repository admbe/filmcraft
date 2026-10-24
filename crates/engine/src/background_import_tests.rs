//! Background imports that cannot finish normally: a worker that ended before opening every
//! file, and a project replaced while its files are still opening.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;

use super::{Pending, poll};
use crate::Session;
use crate::commands::{Prefetched, open_concurrently};

fn write_wav(path: &std::path::Path, seconds: f32) -> String {
    std::fs::write(path, crate::voiceover::write_wav_f32(&vec![0.0; (48_000.0 * seconds) as usize], 48_000)).unwrap();
    path.to_string_lossy().into_owned()
}

/// A worker that died (a panic `guard_job` caught, a thread the OS would not start) leaves some
/// paths without a result. The import used to wait for them forever: it stayed "running", never
/// reported `ImportFinished` and never ran ingest. Now it adds what was opened and says what was not.
#[test]
fn an_import_whose_worker_ended_early_finishes_with_the_files_it_opened() {
    let dir = crate::media_test_util::tmp_dir("background-import-ended");
    let paths: Vec<String> = (0..3).map(|i| write_wav(&dir.join(format!("ended{i}.wav")), 0.5)).collect();
    let mut s = Session::default();
    // the worker opened the first and the last file, then died
    let opened: Arc<Mutex<HashMap<String, Prefetched>>> = Arc::default();
    for p in [&paths[0], &paths[2]] {
        open_concurrently(&s.media, &*s.services, std::slice::from_ref(p), None, &|path: &str, r: Prefetched| {
            opened.lock().unwrap().insert(path.to_string(), r);
        });
    }
    let job = crate::Job { id: 1, label: "Import 3 files".into(), progress: Default::default(), result: Default::default() };
    *job.result.lock().unwrap() = Some(Err("internal error (see the crash log); the job was stopped".into()));
    job.progress.finished.store(true, Ordering::Relaxed);
    s.background_imports.push(Pending {
        job: 1,
        bin: None,
        paths: paths.clone(),
        next: 0,
        opened,
        items: Vec::new(),
        errors: Vec::new(),
        started: web_time::Instant::now(),
        progress: job.progress.clone(),
        result: job.result.clone(),
    });
    s.jobs.push(job);
    poll(&mut s);
    assert!(s.background_imports.is_empty(), "the import is finished, not left running");
    let items = s
        .events
        .iter()
        .find_map(|e| match e {
            crate::Event::ImportFinished { job: 1, items } => Some(items.clone()),
            _ => None,
        })
        .expect("the import reports that it finished");
    let names: Vec<String> = items.iter().map(|i| s.project.item(*i).unwrap().name.clone()).collect();
    assert_eq!(names, ["ended0.wav", "ended2.wav"], "the files that were opened are added, in order");
    assert!(s.log.entries.iter().any(|e| e.message.contains("1 of 3 files not imported")), "the missing file is reported");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Opening another project while a drop of many files is still opening used to add the rest of
/// the files to the other project, one undo step each.
#[test]
fn replacing_the_project_cancels_its_background_imports() {
    let dir = crate::media_test_util::tmp_dir("background-import-replaced");
    let paths: Vec<String> = (0..12).map(|i| write_wav(&dir.join(format!("replaced{i:02}.wav")), 0.5)).collect();
    for replace in [json!(["file.newProject", {"name": "Next"}]), json!(["file.openDemoProject", {}])] {
        let mut s = Session::default();
        let r = s.execute("file.import", json!({ "paths": paths, "async": true })).unwrap();
        let job = r["importJob"].as_u64().unwrap();
        s.execute(replace[0].as_str().unwrap(), replace[1].clone()).unwrap();
        assert!(s.background_imports.is_empty(), "{replace}: the import is dropped with its project");
        let progress = s.jobs.iter().find(|j| j.id == job).unwrap().progress.clone();
        assert!(progress.cancel.load(Ordering::Relaxed), "{replace}: its job is cancelled");
        let started = Instant::now();
        while !progress.finished.load(Ordering::Relaxed) && started.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(progress.finished.load(Ordering::Relaxed), "{replace}: the worker stops");
        for _ in 0..3 {
            poll(&mut s);
        }
        assert!(!s.project.items.values().any(|it| it.name.starts_with("replaced")), "{replace}: nothing reaches the new project");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
