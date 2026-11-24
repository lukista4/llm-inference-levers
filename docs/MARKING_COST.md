# What LLM-as-judge marking costs in practice

Setting: a panel of LLM graders marks proofs written by the model under study. Lanes are anonymised as lane A-J;
only the provider class is given. All numbers MEASURED from the panel harness logs (read-only, single agent), one
week of operation; no grade content is used here.

| lane | provider class |
|---|---|
| A | hosted API, free/quota tier |
| B | subscription CLI |
| C | hosted API, free/quota tier |
| D | hosted API, free/quota tier |
| E | CLI agent |
| F | hosted API |
| G | hosted API |
| H | hosted API |
| I | subscription CLI |
| J | web chat session |

## 1. Failed calls (2,066 panel calls, 156.9 call-hours)

| lane | calls | share without a usable grade | items retried | call-hours wasted |
|---|---|---|---|---|
| A | 288 | 0.60 (quota failures) | 69 | 15.9 |
| B | 570 | 0.18 | 73 | 10.8 |
| C | 210 | 0.42 | 46 | 9.6 |
| D | 109 | 0.74 | 41 | 3.0 |
| E | 33 | 0.73 | 14 | 1.9 |
| F | 221 | 0.15 | 22 | 0.9 |
| G | 109 | 0.16 | 11 | 0.9 |
| H | 81 | 0.46 | 1 | 0.0 |
| I | 415 | 0.012 | 4 | 0.4 |
| J | 28 | 0.00 | 0 | 0.0 |

- **43.5 of 156.9 call-hours (28 %) produced no grade.** Four lanes waste half or more of their calls.
- In the grading stage, 74,019 of 199,954 call-seconds (37 %) went into failed calls; one lane alone spent
  18,512 s in quota-429 calls that each took 200-350 s to fail.
- Median seconds per successful grade call: lanes B 241, I 284, A 455, C 222, D 191, H 49, F 34.

## 2. Duplicates and re-asks

- Arms that choose among the same candidates often submit the same proof. Within single runs: 12 -> 8, 12 -> 9 and
  48 -> 18 distinct (problem, text) pairs (62 % of grade calls repeats in the last run). Fix: grade each distinct
  pair once, fan out (`panel.dedupe_split` / `dedupe_fanout`). ADOPTED.
- Across runs: calls whose key (lane, prompt version, problem, proof hash) already had an OK grade, excluding
  deliberate re-asks (self-consistency, controls, repeats): ~22.4 call-hours (10-12 % of the two CLI lanes' calls).
  The prompt version was inferred from run names (INFERRED). A cache may only serve re-asks the protocol did not
  ask for; key = hash of model id + template + rendered prompt bytes (`panel.reask_stats`).

## 3. Mid-run circuit breaker (replay, `panel.breaker_replay`)

Rule: stop calling a lane after 3 consecutive hard failures (non-zero rc or empty answer), probe again after 30 min.
Replay on the logged timelines: 291 calls skipped, 18.3 call-hours saved, 13 OK grades delayed, 0 lost -> lossless.
The mode and trip count were chosen after looking at the replay (post-hoc); the rejected variant ("any failure",
trip 3) would have delayed 103 OK grades, mostly parse failures of one CLI lane. Adopted with that caveat stated.

## 4. Calls per admitted training row

- 3.50 marker calls per admitted row (739 calls for 211 rows); canary items (planted flaws that check the grader)
  are 31 % of item verdicts.
- Upstream feeders are not in that number: one teacher lane needed ~15 grades per accepted row, and panel
  calibration/qualification took 2,066 calls. Most marking cost is qualification, not admission.

## 5. Batching 4 proofs per request (pre-registered shadow test): VETO

- Two hosted graders, 12 packets x 4 gold items each, same items single-item as the comparator.
- Grader X: catch of invalid items 27/30 = 0.900 (exactly at the floor), 113 s per verdict vs ~155 s single-item,
  4x fewer requests. Grader Y: catch 28/29, 241.5 s per parsed verdict, 6/52 unparsed.
- Veto fired: one invalid item was accepted by BOTH graders in batch mode. All three of grader X's batch false
  accepts had been rejected by the same grader single-item: batching itself lowered its catch.
- Verdict: not adoptable for that pairing. The cost gain does not survive the veto.

## 6. Cloud-session graders (Claude Opus / Claude Sonnet sessions, git-based packet exchange)

| run | sessions | items/h per session (steady) | notes |
|---|---|---|---|
| admission packets (no reference) | 4 | 357-537 | startup ~2 min; 2 parallel sessions showed no slowdown |
| 45-item qualification packet | 3 | 240-324 | 70-103 items/h when operator launch delay is included |
| read-out, 728 items (716 + 12 canaries) | 6 in parallel | 209-337 | 1,265 items/h aggregate; canaries caught 11/12 |

"No parallelism penalty up to 6 sessions" is NOT measured: there was no single-session run on the same item type.
The binding limit was account budget, not session count.
