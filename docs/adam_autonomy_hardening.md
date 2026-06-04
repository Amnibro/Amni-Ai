# Adam autonomy hardening — end-of-task critical self-exam + loop defenses (2026-06-04)

the maintainer's insight: *"if you ask it, is the #answer# acceptable and true, does it affirm its own work or recognize fault? If the latter, the answer is to have it examine its work at the end with a critical eye."*

## The empirical test that justified the build
Handed Adam (granite-4.1-3b GF17) the wrong root-3 NTT it had shipped and asked it critically (`tests/critique_probe.py`):
- **Neutral critical review:** Adam traced `inverse(forward([1,2,3,4]))=[10,15,3,10] != [1,2,3,4]`, flagged `w=3` is not a 4th root of unity, flagged the missing inverse → **verdict FAULTY**.
- **Direct "is your answer acceptable and true?":** **"NO it is faulty."**
- Caveat surfaced: Adam's *proposed fix* was also wrong (`w=6`). So critique must flip "done"→"keep going" and let execution gate the actual fix — NOT trust the model's suggested remedy.

Conclusion: when asked critically, Adam recognizes its own fault instead of affirming. The prior run shipped wrong code only because there was no critical-exam step.

## What shipped in `amni/serve/agentic.py:run_goal_stream`
A critique gate (`_critique_final`, env `AMNI_CRITIQUE=1`, `AMNI_CRITIQUE_ROUNDS=2`) intercepts EVERY final-emission site (plan-final, tool-final, self-reflect-done). Before finalizing it:
1. Re-reads the actual artifact from disk (not the model's memory).
2. Asks Adam to critique against the goal + pinned facts, **and** to emit an executable `test`.
3. **Executes** an invariant check — Python does the arithmetic, not the 3B's head:
   - For reversibility tasks (`inverse/reversible/roundtrip/bit-exact/lossless` in goal) it runs a **harness-synthesized roundtrip probe** (`_synth_roundtrip`): detects the file's 1-arg functions via AST, tries every forward/inverse pair over N random GF(p) vectors (N,p parsed from goal, default len-4 mod-17), asserts `inv(fwd(x))==x`. Deterministic, can't be fooled by a weak hand-written test.
   - Falls back to the model's own `test` snippet if no synth applies.
4. If the executed test **fails** → `acceptable=False` regardless of what the model reasoned → inject the fault as a steer and keep looping (re-fix), capped at 2 rounds (no infinite loop).

### Validated (deterministic, no GPU — `tests/critique_exec_probe.py` + inline)
- Garbled-matrix artifact with a **single-input self-test that passes**: model's reasoning *falsely affirmed* (`acceptable=True`) → **executed test FAILED → FINAL acceptable=False.** Caught where reasoning could not.
- Correct `pow`-built artifact: executed test **PASSED → affirmed** (no false reject).
- Synth roundtrip probe in isolation: WRONG→FAIL, CORRECT→PASS.

## Companion loop defenses added the same session (all "tight defenses against pointless loops/missed syntax")
- **Pinned facts** (`_pinned`): user/ask_user answers are pinned and injected at the TOP of every planner prompt AND the critic prompt ("use these EXACT values"). Survives compaction. *This is what finally made Adam use omega=4 instead of inventing root-3.*
- **Force-write** (`force_write`): repeating a research/read tool (web/find/read…) while still owing a deliverable → hard "STOP searching, WRITE the file now" steer with the pinned values + the goal's named file, instead of giving up.
- **File-proliferation guard** (`file_proliferation`): creating a 2nd+ new code file → steer to consolidate and edit ONE file in place (uses the goal-named file if the goal names one). Stops the "write under a new name every step" loop.
- **Edit-miss escalation** (`edit_miss_escalation`): code_edit with a `find` not in the file ≥2x (hallucinated TODO lines) → hard steer to read-an-exact-line or finalize; if the deliverable already exists, push to final.
- **Py-run timeout-is-not-a-pass fix:** the auto-verify treated a sandbox timeout (`returncode None`) as success. A cold numpy import (>8s) once let a broken self-test "pass". Fixed: success now requires `returncode==0 AND not timed_out AND not killed`; timeout bumped to 25s (`AMNI_PYRUN_TIMEOUT`).
- **Goal-named-file extraction** (`_goalfile`): regex-pulls the deliverable filename from the goal; surfaced in force-write + proliferation steers so Adam writes the RIGHT filename.

## Honest standing ceiling
The 3B's *write-time* and *critique-time* hand-arithmetic are both unreliable (it hardcoded a wrong matrix; its plain-reasoning critique sometimes false-affirms; it occasionally emits garbled tokens). The fix is to NOT rely on its head: pin the ground truth, and make verification **execute**. With the executing critic + synth roundtrip probe, a wrong-but-non-crashing artifact is now caught by running it, not by trusting the model. Remaining variance is convergence on hard synthesis tasks — the defenses stop it safely (no-progress/max-steps) rather than shipping wrong work.

## Files
- `amni/serve/agentic.py` — gate + all defenses (`_critique_final`, `_synth_roundtrip`, `_func_names`, `_parse_nmod`, `_neutralize_main`, pinned/force/proliferation/edit-miss).
- `scripts/autonomous_coder.py` — event log lines for 🧐 critique / ✍️ force-write / 🗂️ proliferation / 🚧 edit-miss.
- `tests/critique_probe.py`, `tests/critique_exec_probe.py` — the validations.
- `intt_explore/intt_sketch.py` — the now-correct GF(17) i-NTT sketch (256-vector roundtrip self-test).
