use std::sync::OnceLock;
use std::thread;
use crate::harness::PythonSandbox;

const N: usize = 256;
const ALPHA: f64 = 0.72;
const BETA_TOOL: f64 = 0.55;
const BETA_ADV: f64 = 0.62;
const BETA_TONE: f64 = 0.28;

fn cents() -> &'static [[f64; 3]; N] {
    static C: OnceLock<[[f64; 3]; N]> = OnceLock::new();
    C.get_or_init(|| {
        let mut p = [[0.0; 3]; N];
        let ga = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for i in 0..N {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / N as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = 2.0 * std::f64::consts::PI * i as f64 / ga;
            let mut v = [r * phi.cos(), r * phi.sin(), z];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v[0] /= n;
            v[1] /= n;
            v[2] /= n;
            p[i] = v;
        }
        p
    })
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn omega_byte(b: u8) -> (f64, f64) {
    let n = N as f64;
    (
        2.0 * std::f64::consts::PI * (b as f64 + 0.5) / n,
        2.0 * std::f64::consts::PI * (b as f64 + 0.5) / (n * 0.5 * (1.0 + 5.0_f64.sqrt())),
    )
}

fn wrap(th: f64, ph: f64) -> (f64, f64) {
    let t = 2.0 * std::f64::consts::PI;
    (th.rem_euclid(t), ph.rem_euclid(t))
}

fn ray(th: f64, ph: f64) -> [f64; 3] {
    let (st, ct) = (th.sin(), th.cos());
    let (sp, cp) = (ph.sin(), ph.cos());
    unit([cp * st, sp * st, ct])
}

fn hit_cell(v: [f64; 3]) -> usize {
    let c = cents();
    let mut best = 0usize;
    let mut sc = f64::NEG_INFINITY;
    for i in 0..N {
        let d = c[i][0] * v[0] + c[i][1] * v[1] + c[i][2] * v[2];
        if d > sc {
            sc = d;
            best = i;
        }
    }
    best
}

pub fn query_state(query: &str) -> (f64, f64, u32, [f64; 3], Vec<u8>) {
    let raw = query.as_bytes().to_vec();
    let mut th = 0.0;
    let mut ph = 0.0;
    let mut st: u32 = 1;
    for &x in &raw {
        let (dth, dph) = omega_byte(x);
        let w = wrap(th + dth, ph + dph);
        th = w.0;
        ph = w.1;
        st = (st.wrapping_shl(1)) ^ (x as u32).wrapping_mul(0x1D);
    }
    (th, ph, st, ray(th, ph), raw)
}

pub fn mix_velocity(v: [f64; 3], f_tool: [f64; 3], f_adv: [f64; 3], f_tone: [f64; 3]) -> [f64; 3] {
    unit(add(
        add(scale(unit(v), ALPHA), scale(unit(f_tool), BETA_TOOL)),
        add(scale(unit(f_adv), BETA_ADV), scale(unit(f_tone), BETA_TONE)),
    ))
}

fn adversary(tape: &[u8]) -> [f64; 3] {
    let has_div = tape.contains(&b'/');
    let has_if = tape.windows(3).any(|w| w == b"if ");
    let has_idx = tape.contains(&b'[');
    let has_len = tape.windows(4).any(|w| w == b"len(");
    let mut f = [0.0; 3];
    if has_div && !has_if {
        f = sub(f, cents()[b'/' as usize]);
    }
    if has_idx && !has_len {
        f = sub(f, cents()[b'[' as usize]);
    }
    unit(f)
}

fn tool(tape: &[u8]) -> [f64; 3] {
    let t = String::from_utf8_lossy(tape);
    if !t.contains("def ") {
        return [0.0; 3];
    }
    let (ok, _o, err, _) = PythonSandbox::execute(&t);
    if ok && err.is_empty() {
        return cents()[hit_cell(ray(0.4, 1.1))];
    }
    if err.contains("ZeroDivision") {
        return unit(add(cents()[b'/' as usize], cents()[b'i' as usize]));
    }
    unit(cents()[b':' as usize])
}

fn tone(tape: &[u8]) -> [f64; 3] {
    if tape.is_empty() {
        return [0.0; 3];
    }
    let punc = tape
        .iter()
        .filter(|b| b"()[]{}:=_.,;".contains(b))
        .count() as f64
        / tape.len() as f64;
    let space = tape.iter().filter(|b| **b == b' ').count() as f64 / tape.len() as f64;
    unit(sub(scale(cents()[b'{' as usize], punc + 0.15), scale(cents()[b' ' as usize], space + 0.05)))
}

#[allow(dead_code)]
pub struct SteerReport {
    pub text: String,
    pub cell: usize,
    pub gf17: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechAct {
    Greeting,
    Status,
    Identity,
    Capabilities,
    Thanks,
    Math,
    Code,
    Debug,
    Web,
    Diff,
    Query,
    Vague,
}

#[derive(Clone, Debug)]
pub struct Classified {
    pub act: SpeechAct,
    pub prototype: String,
    pub affinity: f64,
    pub math_out: Option<String>,
    pub cell: usize,
    pub gf17: u8,
}

const PROTOS: &[(&str, SpeechAct)] = &[
    ("hi", SpeechAct::Greeting),
    ("hello", SpeechAct::Greeting),
    ("hey", SpeechAct::Greeting),
    ("yo", SpeechAct::Greeting),
    ("greetings", SpeechAct::Greeting),
    ("hello adam how are you today", SpeechAct::Status),
    ("how are you", SpeechAct::Status),
    ("how are you today", SpeechAct::Status),
    ("hows it going", SpeechAct::Status),
    ("are you working now", SpeechAct::Status),
    ("are you working", SpeechAct::Status),
    ("are you up", SpeechAct::Status),
    ("are you there", SpeechAct::Status),
    ("are you alive", SpeechAct::Status),
    ("who are you", SpeechAct::Identity),
    ("what are you", SpeechAct::Identity),
    ("what can you do", SpeechAct::Capabilities),
    ("help", SpeechAct::Capabilities),
    ("thanks", SpeechAct::Thanks),
    ("thank you", SpeechAct::Thanks),
    ("sqrt of pi", SpeechAct::Math),
    ("square root of pi", SpeechAct::Math),
    ("what is 2 plus 2", SpeechAct::Math),
    ("compute 3 * 4", SpeechAct::Math),
    ("sin of 0", SpeechAct::Math),
    ("write a python function", SpeechAct::Code),
    ("implement rust code", SpeechAct::Code),
    ("build a shader", SpeechAct::Code),
    ("create a class", SpeechAct::Code),
    ("fix this bug", SpeechAct::Debug),
    ("@debug median", SpeechAct::Debug),
    ("@web https://example.com", SpeechAct::Web),
    ("@diff refactor", SpeechAct::Diff),
    ("what is photosynthesis", SpeechAct::Query),
    ("explain navier stokes", SpeechAct::Query),
];

fn embed_query(query: &str) -> [f64; 3] {
    let n: String = query
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    query_state(n.trim()).3
}

fn proto_rays() -> &'static Vec<([f64; 3], SpeechAct, &'static str)> {
    static P: OnceLock<Vec<([f64; 3], SpeechAct, &'static str)>> = OnceLock::new();
    P.get_or_init(|| PROTOS.iter().map(|(s, a)| (embed_query(s), *a, *s)).collect())
}

fn nearest_act(v: [f64; 3]) -> (SpeechAct, f64, &'static str) {
    let mut best_a = SpeechAct::Vague;
    let mut best_s = f64::NEG_INFINITY;
    let mut best_p = "";
    for (r, act, p) in proto_rays().iter() {
        let d = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
        if d > best_s {
            best_s = d;
            best_a = *act;
            best_p = p;
        }
    }
    (best_a, best_s, best_p)
}

fn is_func_token(t: &str) -> bool {
    matches!(
        t,
        "math.sqrt"
            | "math.sin"
            | "math.cos"
            | "math.tan"
            | "math.log"
            | "math.log10"
            | "math.exp"
            | "math.fabs"
            | "math.factorial"
            | "math.ceil"
            | "math.floor"
    )
}

fn tokenize_math(raw: &str) -> Vec<String> {
    let mut s = raw.to_lowercase();
    for (a, b) in [
        ("what's", "what is"),
        ("whats", "what is"),
        ("√", " sqrt "),
        ("π", " pi "),
        ("×", " * "),
        ("÷", " / "),
        ("^", " ** "),
        ("!", " factorial "),
    ] {
        s = s.replace(a, b);
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '.' || c == '_' {
            buf.push(c);
        } else {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            if "()+-*/,".contains(c) {
                out.push(c.to_string());
            }
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    let mut folded = Vec::new();
    let mut i = 0;
    while i < out.len() {
        if i + 1 < out.len() && out[i] == "square" && out[i + 1] == "root" {
            folded.push("sqrt".into());
            i += 2;
            continue;
        }
        if i + 1 < out.len() && out[i] == "divided" && out[i + 1] == "by" {
            folded.push("/".into());
            i += 2;
            continue;
        }
        folded.push(out[i].clone());
        i += 1;
    }
    folded
}

fn map_math_token(t: &str) -> Option<String> {
    match t {
        "sqrt" | "root" => Some("math.sqrt".into()),
        "sin" => Some("math.sin".into()),
        "cos" => Some("math.cos".into()),
        "tan" => Some("math.tan".into()),
        "log" | "log10" => Some("math.log10".into()),
        "ln" => Some("math.log".into()),
        "exp" => Some("math.exp".into()),
        "abs" => Some("math.fabs".into()),
        "factorial" => Some("math.factorial".into()),
        "ceil" => Some("math.ceil".into()),
        "floor" => Some("math.floor".into()),
        "pi" => Some("math.pi".into()),
        "e" | "euler" => Some("math.e".into()),
        "tau" => Some("math.tau".into()),
        "plus" => Some("+".into()),
        "minus" => Some("-".into()),
        "times" | "multiplied" => Some("*".into()),
        "over" => Some("/".into()),
        "mod" | "modulo" => Some("%".into()),
        "to" | "power" => Some("**".into()),
        _ => None,
    }
}

const DROP: &[&str] = &[
        "what", "is", "the", "a", "an", "of", "please", "me", "can", "you", "could", "would",
        "tell", "give", "value", "number", "equals", "equal", "for", "compute", "calculate",
        "find", "evaluate", "just", "quick", "need", "want", "i", "we", "my", "your", "be",
        "it", "this", "that", "answer",
];

fn compile_math(query: &str) -> Option<String> {
    let raw = tokenize_math(query);
    if raw.is_empty() {
        return None;
    }
    let mut toks: Vec<String> = Vec::new();
    for t in raw {
        if DROP.contains(&t.as_str()) {
            continue;
        }
        if let Some(m) = map_math_token(&t) {
            toks.push(m);
            continue;
        }
        if t.chars().all(|c| c.is_ascii_digit() || c == '.') || "()+-*/,%".contains(t.as_str()) || t == "**" {
            toks.push(t);
            continue;
        }
        if t.starts_with("math.") {
            toks.push(t);
            continue;
        }
        return None;
    }
    if toks.is_empty() {
        return None;
    }
    let has_num_or_const = toks.iter().any(|t| {
        t.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
            || t == "math.pi"
            || t == "math.e"
            || t == "math.tau"
    });
    if !has_num_or_const && !toks.iter().any(|t| is_func_token(t)) {
        return None;
    }
    let mut assembled = String::new();
    let mut i = 0;
    while i < toks.len() {
        if is_func_token(&toks[i]) {
            let fname = toks[i].clone();
            i += 1;
            if i < toks.len() && toks[i] == "(" {
                assembled.push_str(&fname);
                continue;
            }
            let mut depth = 0;
            let mut arg = String::new();
            while i < toks.len() {
                let t = &toks[i];
                if is_func_token(t) && depth == 0 && !arg.is_empty() {
                    break;
                }
                if t == "(" {
                    depth += 1;
                }
                if t == ")" {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                if "+-*/%".contains(t.as_str()) || t == "**" {
                    if depth == 0 && !arg.is_empty() {
                        break;
                    }
                }
                arg.push_str(t);
                i += 1;
                if depth == 0 && !arg.is_empty() && (i >= toks.len() || is_func_token(&toks[i]) || "+-*/%".contains(toks[i].as_str()) || toks[i] == "**" || toks[i] == ")") {
                    break;
                }
            }
            if arg.is_empty() {
                return None;
            }
            assembled.push_str(&format!("{}({})", fname, arg));
            continue;
        }
        assembled.push_str(&toks[i]);
        i += 1;
    }
    if assembled.is_empty() {
        return None;
    }
    let ok = assembled.chars().all(|c| {
        c.is_ascii_alphanumeric() || "._+-*/()%, ".contains(c)
    });
    if !ok {
        return None;
    }
    Some(assembled)
}

fn eval_math(expr: &str) -> Option<String> {
    let code = format!("import math\nprint(repr({}))", expr);
    let (ok, out, err, _) = PythonSandbox::execute(&code);
    if !ok || !err.is_empty() {
        return None;
    }
    let v = out.trim();
    if v.is_empty() {
        return None;
    }
    Some(v.to_string())
}

fn interrogative(q: &str) -> bool {
    let t = q.trim().to_lowercase();
    t.contains('?')
        || t.starts_with("what")
        || t.starts_with("who")
        || t.starts_with("how")
        || t.starts_with("why")
        || t.starts_with("are ")
        || t.starts_with("is ")
        || t.starts_with("do ")
        || t.starts_with("can ")
        || t.starts_with("does ")
}

fn imperative(q: &str) -> bool {
    let t = q.to_lowercase();
    ["write ", "implement ", "build ", "create ", "make ", "fix ", "refactor "]
        .iter()
        .any(|v| t.contains(v))
}

fn math_charset(q: &str) -> bool {
    q.chars().any(|c| c.is_ascii_digit() || "+-*/^=√π".contains(c))
        || ["sqrt", "sin", "cos", "tan", "pi", "log", "ln"].iter().any(|w| {
            q.to_lowercase().split(|c: char| !c.is_alphanumeric()).any(|t| t == *w)
        })
}

fn intent_forces(query: &str) -> ([f64; 3], [f64; 3], [f64; 3], Option<String>) {
    let q = query.to_lowercase();
    let math_expr = compile_math(&q);
    let math_val = math_expr.as_ref().and_then(|e| eval_math(e));
    let mut f_tool = [0.0; 3];
    if math_val.is_some() {
        f_tool = query_state("sqrt of pi").3;
    }
    let mut f_adv = [0.0; 3];
    if interrogative(&q) && !imperative(&q) {
        f_adv = unit(sub([0.0; 3], query_state("write a python function").3));
    }
    if math_charset(&q) && !imperative(&q) {
        f_adv = add(f_adv, unit(sub([0.0; 3], query_state("build a shader").3)));
        f_tool = add(f_tool, query_state("compute 3 * 4").3);
    }
    let f_tone = if q.contains('?') {
        query_state("how are you").3
    } else if q.split_whitespace().count() <= 2 {
        query_state("hi").3
    } else {
        tone(q.as_bytes())
    };
    (unit(f_tool), unit(f_adv), unit(f_tone), math_val)
}

pub fn classify(query: &str) -> Classified {
    let v = embed_query(query);
    let q = query.to_string();
    let handle = thread::spawn(move || intent_forces(&q));
    let (f_tool, f_adv, f_tone, math_out) = handle.join().unwrap_or(([0.0; 3], [0.0; 3], [0.0; 3], None));
    let steered = mix_velocity(v, f_tool, f_adv, f_tone);
    let (raw_act, raw_aff, raw_proto) = nearest_act(v);
    let (mut act, mut aff, mut proto) = nearest_act(steered);
    if math_out.is_some() && !imperative(query) {
        act = SpeechAct::Math;
        proto = "math_sandbox";
        aff = aff.max(0.8);
    } else if matches!(
        raw_act,
        SpeechAct::Greeting | SpeechAct::Status | SpeechAct::Identity | SpeechAct::Thanks | SpeechAct::Capabilities
    ) && math_out.is_none()
    {
        act = raw_act;
        aff = raw_aff;
        proto = raw_proto;
    } else if aff < 0.22 {
        act = SpeechAct::Vague;
    }
    let cell = hit_cell(steered);
    Classified {
        act,
        prototype: proto.to_string(),
        affinity: aff,
        math_out,
        cell,
        gf17: (cell % 17) as u8,
    }
}

fn gf_inv(pick: usize, st: u32) -> u8 {
    let a = ((3u32 ^ (st & 255)) | 1) as i32;
    let ai = mod_inverse_odd(a, 256);
    let y = pick as i32;
    ((((y - 17) * ai) % 256 + 256) % 256) as u8
}

fn mod_inverse_odd(a: i32, m: i32) -> i32 {
    let (mut t, mut newt) = (0i32, 1i32);
    let (mut r, mut newr) = (m, a.rem_euclid(m));
    while newr != 0 {
        let q = r / newr;
        (t, newt) = (newt, t - q * newt);
        (r, newr) = (newr, r - q * newr);
    }
    t.rem_euclid(m)
}

fn delim_stack(bytes: &[u8]) -> Vec<u8> {
    let mut s = Vec::new();
    for &b in bytes {
        match b {
            b'(' => s.push(b')'),
            b'[' => s.push(b']'),
            b'{' => s.push(b'}'),
            b'"' | b'\'' => {
                if s.last() == Some(&b) {
                    s.pop();
                } else {
                    s.push(b);
                }
            }
            _ => {
                if s.last() == Some(&b) {
                    s.pop();
                }
            }
        }
    }
    s
}

fn force_window<'a>(seed: &'a [u8], tape: &'a [u8]) -> &'a [u8] {
    if seed.windows(4).any(|w| w == b"def ") {
        return seed;
    }
    tape
}

fn seed_closed(seed: &[u8], stack: &[u8]) -> bool {
    if !stack.is_empty() {
        return false;
    }
    let t = String::from_utf8_lossy(seed);
    t.contains("def ") && t.contains('\n') && (t.contains("return ") || t.ends_with('\n'))
}

pub fn run(query: &str, steps: usize) -> SteerReport {
    let (mut th, mut ph, mut st, mut v, seed) = query_state(query);
    let mut tape = seed.clone();
    let mut stack = delim_stack(&seed);
    let mut walked = 0usize;
    let mut indent_pending = 0u8;
    if stack.is_empty() && seed_closed(&seed, &stack) {
        let win = force_window(&seed, &tape);
        let t_adv = win.to_vec();
        let t_tool = win.to_vec();
        let t_tone = win.to_vec();
        let h_adv = thread::spawn(move || adversary(&t_adv));
        let h_tool = thread::spawn(move || tool(&t_tool));
        let h_tone = thread::spawn(move || tone(&t_tone));
        v = mix_velocity(
            v,
            h_tool.join().unwrap_or([0.0; 3]),
            h_adv.join().unwrap_or([0.0; 3]),
            h_tone.join().unwrap_or([0.0; 3]),
        );
        let cell = hit_cell(v);
        return SteerReport {
            text: String::from_utf8_lossy(&seed).into_owned(),
            cell,
            gf17: (cell % 17) as u8,
        };
    }
    for t in 0..steps {
        let bt = if indent_pending > 0 {
            indent_pending -= 1;
            32u8
        } else {
            let cell = hit_cell(v);
            gf_inv(cell, st)
        };
        tape.push(bt);
        stack = delim_stack(&tape);
        let (dth, dph) = omega_byte(bt);
        let w = wrap(th + dth, ph + dph);
        th = w.0;
        ph = w.1;
        st = st.wrapping_shl(1) ^ (bt as u32).wrapping_mul(0x1D);
        v = ray(th, ph);
        walked += 1;
        if bt == b'\n' {
            let prev = tape.iter().rev().skip(1).copied().find(|x| *x != b'\n' && *x != b' ');
            if prev == Some(b':') {
                indent_pending = 4;
            }
        }
        if t % 4 == 0 {
            let win = force_window(&seed, &tape).to_vec();
            let t_adv = win.clone();
            let t_tool = win.clone();
            let t_tone = win;
            let h_adv = thread::spawn(move || adversary(&t_adv));
            let h_tool = thread::spawn(move || tool(&t_tool));
            let h_tone = thread::spawn(move || tone(&t_tone));
            v = mix_velocity(
                v,
                h_tool.join().unwrap_or([0.0; 3]),
                h_adv.join().unwrap_or([0.0; 3]),
                h_tone.join().unwrap_or([0.0; 3]),
            );
        }
        let last = *tape.last().unwrap_or(&32);
        if stack.is_empty() && walked >= 8 && matches!(last, b'.' | b'?' | b'!') {
            break;
        }
    }
    let emit = if tape.len() > seed.len() {
        String::from_utf8_lossy(&tape[seed.len()..]).into_owned()
    } else {
        String::from_utf8_lossy(&seed).into_owned()
    };
    let cell = hit_cell(v);
    SteerReport {
        text: emit,
        cell,
        gf17: (cell % 17) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mix_stays_on_sphere() {
        let v = mix_velocity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.2, 0.2]);
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-9);
    }
    #[test]
    fn query_phase_differs() {
        let a = query_state("abc");
        let b = query_state("abd");
        assert!((a.0 - b.0).abs() > 1e-9);
    }
    #[test]
    fn compile_sqrt_pi() {
        let e = compile_math("what's the sqrt of pi").expect("expr");
        assert!(e.contains("math.sqrt") && e.contains("math.pi"));
        let v = eval_math(&e).expect("val");
        let n: f64 = v.parse().unwrap();
        assert!((n - std::f64::consts::PI.sqrt()).abs() < 1e-9);
    }
    #[test]
    fn classify_status_not_code() {
        let v = embed_query("are you working now?");
        let (raw, aff, proto) = nearest_act(v);
        assert_eq!((raw, proto, aff > 0.0), (SpeechAct::Status, "are you working now", true), "raw nn failed aff={aff} proto={proto}");
        let c = classify("are you working now?");
        assert_eq!(c.act, SpeechAct::Status, "steered proto={}", c.prototype);
        assert!(c.math_out.is_none());
    }
    #[test]
    fn classify_math_sqrt_pi() {
        let c = classify("what's the sqrt of pi");
        assert_eq!(c.act, SpeechAct::Math);
        assert!(c.math_out.is_some());
    }
    #[test]
    fn classify_hi_greeting() {
        let c = classify("hi");
        assert_eq!(c.act, SpeechAct::Greeting);
    }
    #[test]
    fn closed_def_has_no_emit_tail() {
        let src = "def g(a,b):\n    return a/b\n";
        let r = run(src, 40);
        assert_eq!(r.text, src);
    }
}
