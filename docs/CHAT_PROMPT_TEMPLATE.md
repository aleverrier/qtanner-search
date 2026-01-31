# Prompt template for a new chat

Use this as a ready-to-paste prompt for a new Codex/ChatGPT session.

```
You are working in repo: /Users/anthony/research/qtanner-search-maint
Current date: 2026-01-31

Project: explicit quantum Tanner codes from left–right Cayley complexes (LRZ, arXiv:2512.20532).

Two-machine workflow:
- Search runs on one clone.
- Publishing happens only on a maintenance clone.
- Archive untracked artifacts to local_results/ before git pull/rebase.

Constraints:
- Do NOT run anything that takes more than a few minutes on a MacBook Pro.
- Prefer filter-first workflows; no day-long searches.
- Use ./scripts/run_tests.sh (or make test) for checks.
- All matrices saved in MatrixMarket .mtx format.

Current tracks:
- 633 track (default): local [6,3,3], publishes to best_codes/.
- 844 track: local [8,4,4], publishes to best_codes_844/ and pending artifacts to codes/pending_844/.

Next objective:
- Continue the [8,4,4] track: refine/publish, add analysis, or extend progressive search if needed.
- Ensure the 844 best-codes table remains separate and the website data is consistent.

Start by reading docs/CHAT_CONTEXT.md and docs/LOCAL_CODES_844_PLAN.md.
```
