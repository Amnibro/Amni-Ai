use adam_ks::{crc16, jstr, paginate, pyjson, store::{dir_pack, dir_unpack}};
use serde_json::json;
#[test]
fn json_matches_python() {
    assert_eq!(jstr("Ünïcödé ✓ 漢字 🚀 \"q\" \\ \t\u{7f}\u{1} end"), r#""\u00dcn\u00efc\u00f6d\u00e9 \u2713 \u6f22\u5b57 \ud83d\ude80 \"q\" \\ \t\u007f\u0001 end""#);
    assert_eq!(pyjson(&json!({"a": [1, {"b": []}], "c": {}, "d": "é"}), Some(1), 0), "{\n \"a\": [\n  1,\n  {\n   \"b\": []\n  }\n ],\n \"c\": {},\n \"d\": \"\\u00e9\"\n}");
    assert_eq!(pyjson(&json!({"n": 3, "atlas": [1, 4096, 4]}), None, 0), "{\"n\": 3, \"atlas\": [1, 4096, 4]}");
}
#[test]
fn directory_matches_python() {
    let d = dir_pack(&[0, 8200, 17], &[0, 4096, 5], &[0, 65535, 12345]);
    assert_eq!(d, vec![[0; 12], [6, 6, 11, 1, 16, 2, 14, 6, 0, 13, 5, 13], [0, 1, 0, 0, 5, 0, 0, 11, 3, 12, 8, 2]]);
    assert_eq!(dir_unpack(&d).unwrap(), (vec![0, 8200, 17], vec![0, 4096, 5], vec![0, 65535, 12345]));
    let mut bad = d.clone();
    bad[1][3] = (bad[1][3] + 1) % 17;
    assert!(dir_unpack(&bad).is_err());
    bad[1][3] = 17;
    assert!(dir_unpack(&bad).is_err());
}
#[test]
fn paginate_matches_python() {
    let b = [b"word ".repeat(1000), b"\n\n".to_vec(), b"x".repeat(5000), "é".repeat(3000).into_bytes()].concat();
    let p = paginate(&b);
    assert_eq!(p.iter().map(|x| x.len()).collect::<Vec<_>>(), vec![4095, 4096, 4095, 3716]);
    assert_eq!((p.concat(), crc16(&b)), (b.clone(), 1171));
    assert_eq!(paginate(b""), vec![&b""[..]]);
}
