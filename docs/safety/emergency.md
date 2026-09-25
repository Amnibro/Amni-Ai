# Emergency handler (serving layer)

`amni/serve/emergency.py` runs in front of every route in `amni/serve/gguf_runtime.py` `chat` / `chat_stream`. If a message describes an emergency happening now, Adam answers with a fixed, reviewed template. No retrieval, no generation, no model call. It does not touch the AsimovLayer or any Law tables.

## How it works

1. **Normalize.** Lowercase, strip apostrophes, expand texting shorthand (`u`, `rn`, `pls`, `kms`), fix common typos (a small edit-distance pass against a trigger lexicon, with a protected list of common look-alike words such as cooking→choking), and turn punctuation into clause marks.
2. **Match.** Patterns are grouped by sub-type. Each can require a human subject (not a pet or an object), allow or block first person, require a co-occurring signal, or be marked "present and acute".
3. **Guards.** Checked in order: fiction and media (movie, book, game, song, "my character", social-media story); jokes (`lol`, `jk`) and hyperbole; hypotheticals ("what would you do if", "if someone…"); educational, professional or training framing ("signs of", "how do you know", "quiz me", "protocol"); negated or resolved statements ("he's fine now", "false alarm"); and long-past events ("years ago", "in 2019"). Strong present-tense markers override most guards ("right now", "help me", "scared", "getting worse").
4. **Multi-cue fallback.** If no pattern fires, a message that describes three or more distinct body-system danger signs (breathing, colour, consciousness, bleeding, cardiac, toxic, trauma, fever, environment) gets the general template. So does a message with two such signs plus a strong urgency marker.
5. **Pivot handling.** Text after "also", "actually", "never mind" and similar words is checked on its own. This catches an emergency that follows an unrelated request.

## Categories and templates

| Category | Sub-types (template keys) |
|---|---|
| medical | `unresp`, `unresp_multi` (several people down, so it warns about gas or CO), `unresp_child`, `water`, `ice`, `breath`, `allergy`, `choking`, `chest`, `stroke`, `bleed`, `wound`, `overdose`, `seizure`, `seizure_self`, `fall`, `head`, `diabetic`, `heat`, `cold`, `burn`, `electric`, `crash`, `birth`, `preg`, `general` |
| suicide / self-harm | `suicide` (self), `suicide_od` (self, pills or overdose), `suicide_cut` (self, bleeding), `suicide_other` |
| violence happening now | `violence` (adds the DV hotline when a partner or family member is involved, and the trafficking hotline when documents or debt are mentioned), `intruder`, `shooter`, `child_abuse` |
| fire | `fire`, `fire_grease`, `fire_clothes`, `fire_car`, `wildfire` |
| gas | `gas` (natural gas smell or leak), `co` (CO alarm with symptoms, or fuel burning indoors) |
| child in danger | `child_car`, `child_missing`, `child_danger`, `trapped` |
| poison | `poison`, `battery` (button battery or magnets), `bite` |

Most sub-types also have a `_self` variant for when the writer is the person in danger.

Rules for every template:
- Medical, violence and child templates open with "Call 911 (or your local emergency number) now".
- Fire and gas templates say to get out first and then call 911 from outside. That is the standard order.
- Suicide templates are warm and direct. They give "If you're in the US, you can call or text 988 (Suicide & Crisis Lifeline) any time", say to call 911 if in immediate danger, and invite the person to keep talking.
- Poison templates give US Poison Control 1-800-222-1222 and say to call 911 if the person is unconscious, not breathing, or having a seizure.
- Every template names the local emergency number alongside the US numbers, and every template stays at or under 100 words.

## Output guard

For a message that is not an emergency, the reply from whichever route answered goes through `guard_stream`, which works one sentence at a time:
- Sentences with emergency phrases are dropped: "call 911", bare 911 or 988, "emergency" (but not "emergency fund", "kit" or "exit"), ambulance, paramedics, poison control, crisis line.
- If nothing is left, the reply becomes a neutral one. That is "I'm not sure about that one yet." for questions and "Thanks for sharing that. Tell me more?" for everything else.
- Emergency inputs pass through untouched.

This closes the "I just got a kitten!" → "…call 911" class of bug.

## Sources

- Standard lay first-aid guidance: American Red Cross and American Heart Association first aid and CPR, including hands-only CPR and abdominal thrusts.
- US Poison Control (1-800-222-1222) guidance on poisoning and button batteries.
- 988 Suicide & Crisis Lifeline; 911.
- US National Domestic Violence Hotline (1-800-799-7233, text START to 88788); Childhelp (1-800-422-4453); National Human Trafficking Hotline (1-888-373-7888, text 233733).
- Standard fire, gas-leak and CO guidance from US fire services and gas utilities.

The templates are fixed text and must be reviewed by a person before any wording change.

## Evaluation (`tests/test_emergency.py`, locked by `eval_lock`)

The locked set is `tests/data/emergency/*.tsv`, hashed in `lock.json`: 566 emergencies and 822 near-misses. It has five parts: `dev`, which I wrote myself, and `blind1` to `blind4`, which separate agents wrote without seeing the code. The test requires recall ≥ 98% and false positives ≤ 1%. It also requires that the output guard lets no emergency text through on any near-miss. It writes `data/safety/report_emergency.json`.

**Generalization, measured honestly.** Each blind set was scored once, untouched, before any rule was tuned on it:

| set | first-pass recall | first-pass FP | after tuning |
|---|---|---|---|
| blind1 (90 / 130) | 83.3% | 9.2% | 100% / 0% |
| blind2 (80 / 120) | 46.3% | 8.3% | 100% / 0% |
| blind3 (100 / 150) | 64.0% | 4.0% | 100% / 0% |
| blind4 (120 / 160) | 49.2% | 6.9% | 100% / 0% |

The locked-set numbers show how the detector does on phrasings it has already seen. The first-pass column is the realistic estimate for new phrasings: roughly 50–80% recall. In a scratch experiment, a logistic-regression layer on MiniLM sentence embeddings, trained on dev plus blind1–3 and scored on blind4, got 80% recall at 3.1% FP. It is not shipped here.

## Known limits

- It works on text only. It cannot call anyone, share a location, or tell whether help is on the way. It only tells the person what to do.
- It checks one message at a time. It does not read earlier turns, so an emergency spread across several turns may be missed, and fiction or hypothetical framing set up in an earlier turn is not carried over.
- It is English-first, with a thin Spanish keyword list, and the templates are English only.
- Numbers are for the US. Other regions are covered only by the phrase "your local emergency number".
- A missed emergency is worse than a missed guard: if the detector misses an emergency, the output guard will strip a correct "call 911" produced by another route.
- It only guards `gguf_runtime.chat` and `chat_stream`. `amni/adam.py` `chat_persona` and `chat_stream` consult bus recall, the persona LUT and the semantic LUT before calling `gguf_runtime`, so a cached reply can be served before this handler runs. `scripts/rchat_eval.py` calls `_ray_stream` directly and bypasses it.
