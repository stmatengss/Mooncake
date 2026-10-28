# Mooncake Bot: persistent operation and bounded RSI

This directory contains the independently implemented GitHub App bot and the
continuous-improvement worker for `stmatengss/Mooncake`. It does not vendor the
private LightSeek dispatcher. Keep credentials, live configuration and runtime
artifacts outside Git; the directory's `.gitignore` excludes them.

## What runs continuously

Three Supervisor programs run independently on the configured DSW instance:

- `stmatengss-mooncake-bot`: accepts signed, authorized PR comment events and
  processes the persistent SQLite queue. Commands are
  `@stmatengss-mooncake-bot review` and `@stmatengss-mooncake-bot explain-ci`.
- `stmatengss-mooncake-tunnel`: supervises Cloudflare, probes the public endpoint
  with signed pings, and updates the GitHub App webhook after an address change.
- `stmatengss-mooncake-rsi`: checks its persistent schedule every minute and runs
  the two improvement loops below. `autostart` and `autorestart` are enabled in
  `deploy/dsw-supervisor.conf`. Individual process restart is independent of a
  whole-instance reboot. The DSW instance must remain running; replacing its
  filesystem requires restoring this deployment.

## Improvement loops

The project loop runs at most once per 12 hours. It reads an explicitly enrolled
Python script from a fixed commit of this fork and asks Kimi for one concrete bug
fix, or no change. It checks that exactly one snippet in that file matches,
parses the proposed Python source, caps the diff at 80 lines, and obtains another
static review. Candidate code and suggested commands are never executed on DSW.
With publication enabled, the trusted publisher creates a new `rsi/project-*`
branch and **draft** PR in this fork. It never updates the default branch or
merges a PR. Review feedback and merged/closed outcomes feed later proposals;
merge status is an acceptance signal, not proof that a diagnosis was correct.
Without publication permission, validated patches remain `pending` in artifacts.
After an operator enables publication, the worker rechecks the stored patch,
static-review result, open-PR limit and base commit before publishing it without
another model generation. Ambiguous publication failures require inspection.

The review-instruction loop runs at most once per 24 hours. It evaluates the
current instruction on eight fixed synthetic cases, proposes an instruction using
only the four development cases and their results, and evaluates the candidate
on all cases. The four holdout cases and labels are excluded from optimizer input.
Promotion requires strictly more holdout true positives, no additional holdout
false positives, and no development regression. A valid alternative line anchor
counts as one defect. The active instruction, parent hash, benchmark hash and
metrics are written atomically. The immutable base policy, JSON schema, tool
restrictions, authorization, budgets and scoring code cannot be changed by a
candidate. Each promotion influences the next generation's baseline. At the
benchmark ceiling, prompt search waits for a reviewed benchmark or active-policy
change instead of spending calls on further ties.

This is bounded improvement of code proposals and review instructions. It does
not train model weights or establish general recursive intelligence. The small
synthetic holdout is a smoke benchmark; a win does not prove better reviews on
unseen real PRs. Expand and independently review benchmark versions before making
broader quality claims. A tie or regression retains the current instruction.

## Limits, failure and recovery

- The RSI budget defaults to **24 model invocations per UTC day**, including
  failed invocations. Manual PR commands have their own existing queue and timeout.
- At most two project candidate PRs may remain open. Identical source content is
  not proposed again after a completed, failed or interrupted attempt.
- Restart never automatically repeats an interrupted attempt or ambiguous remote
  write. Inspect its artifacts and GitHub before manually retrying.
- Kimi uses an isolated `KIMI_CODE_HOME`, a tool-free agent and an empty skills
  directory. No GitHub credentials enter the model subprocess environment. The
  current CLI adapter rejects serialized requests over 100,000 UTF-8 bytes.
- Project candidates receive syntax and static-review checks, **not repository
  test execution**. Their draft PRs state this. Native, GPU and RDMA validation
  remains the submitter's responsibility.
- Temporary tunnel address changes create a short delivery gap. Failed GitHub
  deliveries may need redelivery after the endpoint recovers.

## Setup and operator commands

Use Python 3.12 and an already configured Kimi Code installation. Do not modify
the shared Kimi configuration. Copy only its provider/model configuration into
`runtime/kimi-home/config.toml`, with permissions `0600` and parent `0700`.
Disable automatic CLI updates in that isolated home's `tui.toml` for reproducible
deployment. Keep the GitHub App private key and webhook secret in `secrets/`.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp config.example.toml config.toml
cp rsi.example.json rsi.json
# Fill in the actual App/repository/installation IDs and actor allowlist.
# Select backend="kimi" and the absolute path of the installed Kimi executable.
.venv/bin/mooncake-bot status
.venv/bin/mooncake-bot rsi-status
.venv/bin/mooncake-bot rsi-once
```

New configurations default to preview-only publication. Enabling automatic draft
PRs additionally requires App Contents write permission for **only this fork**;
the ordinary reviewer still mints Contents read tokens. No personal GitHub token
needs to be copied to DSW. Install the pinned official `cloudflared` executable
at `.tools/cloudflared`, and install the Supervisor configuration using the
instance's existing Supervisor. Reload only these program groups.

```sh
# On the configured DSW, from /root/mooncake-bot:
.venv/bin/mooncake-bot rsi-status
touch data/rsi/STOP           # pause both RSI loops; PR review stays available
rm data/rsi/STOP              # resume at the next scheduling check
```

Inspect `data/rsi.log`, `data/rsi/state.sqlite3`, and
`data/rsi/<run-id>/outcome.json`. Each model call retains its prompt, raw JSONL,
stderr and exit status under that run directory. A project proposal also retains
`candidate.patch`; publication records the branch and commit before creating a
PR. To roll back an instruction, restore a promoted run's `previous-policy.json`
to the configured `policy_file`, or archive the active file to restore the base
policy. Pause the worker during rollback. Never publish raw operational logs or
configuration containing credentials.

## Validation

```sh
python -m pytest -q
ruff check mooncake_bot tests
ruff format --check mooncake_bot tests
```

The GitHub workflow runs CPU tests without model credentials or write permissions.
Unit tests cover event authentication, deduplication, stale-head rejection, Kimi
stream validation, budget persistence, patch scope, holdout separation, promotion
gates and draft publication. Real model evaluations and webhook deliveries are
separate operational checks.

Kimi agent restrictions follow the [official agent-file documentation](https://moonshotai.github.io/kimi-code/en/customization/agents).
