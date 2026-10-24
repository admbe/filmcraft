//! A card's worth of files imported in one command: they are opened several at a time (opening
//! is the slow part; some decoders take a tenth of a second) but added in the order given,
//! with one error per file that could not be imported and the rest imported anyway.

use serde_json::json;

use crate::Session;
use filmcraft_project::ItemId;

fn write_wav(path: &std::path::Path, seconds: f32) -> String {
    std::fs::write(path, crate::voiceover::write_wav_f32(&vec![0.0; (48_000.0 * seconds) as usize], 48_000)).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn a_batch_import_keeps_the_given_order_and_reports_each_failure() {
    let dir = crate::media_test_util::tmp_dir("import-batch");
    // different lengths, so each item can be matched to its file
    let mut paths: Vec<String> = (0..12).map(|i| write_wav(&dir.join(format!("clip{i:02}.wav")), 0.25 + i as f32 * 0.25)).collect();
    let missing = dir.join("not-there.wav").to_string_lossy().into_owned();
    paths.insert(5, missing.clone());
    let mut s = Session::default();
    let r = s.execute("file.import", json!({ "paths": paths })).unwrap();
    let items: Vec<u64> = r["items"].as_array().unwrap().iter().filter_map(|v| v.as_u64()).collect();
    assert_eq!(items.len(), 12, "every readable file imported: {r}");
    let errors = r["errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1, "one error, for the missing file: {r}");
    assert!(errors[0].as_str().unwrap().contains("not-there.wav"));
    let names: Vec<String> = items.iter().map(|i| s.project.item(ItemId(*i)).unwrap().name.clone()).collect();
    let expected: Vec<String> = (0..12).map(|i| format!("clip{i:02}.wav")).collect();
    assert_eq!(names, expected, "items in the order the paths were given");
    for (i, id) in items.iter().enumerate() {
        let d = s.project.item(ItemId(*id)).unwrap().duration().seconds();
        assert!((d - (0.25 + i as f64 * 0.25)).abs() < 0.01, "item {i} has its own file's duration ({d} s)");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
