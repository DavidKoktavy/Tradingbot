# trading_agent

A modular, code-driven trading system for Interactive Brokers, with AI used
for research/orchestration and a **deterministic risk boundary** that AI
output can never cross.

## Status

Under incremental development, phase by phase. See "Roadmap" below.
**No component in this repository is authorized to place live orders yet** —
`TRADING_MODE` defaults to `PAPER`, and reaching `LIVE` requires two
independent, explicitly-set environment values checked at startup
(`app/config.py::Settings._guard_live_trading`).

## Safety model (read this first)

- **AI proposes, code disposes.** The AI decision layer returns a
  schema-validated `OrderIntent` candidate. It is treated as untrusted
  input and passed through the same `RiskEngine` and `OrderValidator` as
  any rule-based strategy's intent. There is no code path from an AI
  response directly to `IBKRClient`.
- **PAPER is the default and BACKTEST/SIMULATION/PAPER/LIVE is a strict
  progression.** Nothing auto-promotes to LIVE because backtests or paper
  results looked good.
- **Fail closed.** On broker disconnect, stale data, DB outage, or AI
  timeout, the system stops opening new positions and reconciles state
  rather than guessing.
- Backtest results are not evidence of future profitability. No strategy
  shipped here is presented as profitable.

## Requirements

- Python 3.12+
- PostgreSQL (production; SQLite acceptable for local dev/tests later)
- TWS or IB Gateway running locally with API access enabled, for any
  phase beyond backtesting

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env          # never commit real credentials
pytest                        # 496 tests, no network required

python -m trading_agent status
python -m trading_agent strategies
python -m trading_agent backtest --strategy ma_crossover --data bars.csv
python -m trading_agent explain --audit-log logs/decisions.jsonl
python -m trading_agent dashboard        # http://127.0.0.1:8000
```

There is deliberately **no `live` command**. Running live requires setting
`TRADING_MODE=LIVE`, `ENABLE_LIVE_TRADING=true`, and the exact
`LIVE_TRADING_CONFIRMATION` phrase — a considered act performed outside
the convenience of a CLI flag.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # edit values; NEVER commit real credentials
pytest
```

## Roadmap

| Phase | Scope |
|---|---|
| 1 | Architecture, repo structure, typed configuration ✅ |
| 2 | Market data abstraction + IBKR connection ✅ |
| 3 | Portfolio, positions, order management ✅ |
| 4 | Risk engine ✅ (this commit) |
| 5 | Strategy framework + example strategies |
| 6 | Backtesting engine |
| 7 | AI decision layer |
| 8 | Paper trading |
| 9 | Monitoring + dashboard |
| 10 | Hardening, testing, deployment |

## Autonomous order submission notes

A user asked directly "how do I make it place orders itself?" and the
honest answer, at the time, was: nothing actually did. Two real gaps,
found by trying to answer the question rather than by inspection.

**Gap 1 — `paper` never touched IBKR at all.** `_run_loop` was shared
between `simulate` and `paper` and literally printed *"NOTE: no market
data source is attached in this build"* every time, regardless of mode.
Fixed:

- `paper` mode now constructs a real `IBKRClient`, connects it, builds a
  `LiveMarketDataFeed` from it, and swaps in a real `IBKROrderGateway` in
  place of the default `SimulatedBrokerGateway` — so orders the loop
  decides to submit actually reach your IBKR paper account.
- Connection failures and market-data warm-up failures (`feed.start()`
  raising because there's no historical data) are caught explicitly,
  reported in plain language pointing back at `smoke_test_ibkr.py`, and
  always disconnect the client cleanly — never a raw traceback, never an
  abandoned TWS API session.
- An unmissable line prints before the loop starts: *"This is connected
  to your REAL IBKR paper account. It WILL decide and submit orders
  autonomously against it, using pretend money."* No one should be
  surprised that autonomous submission has begun.
- The mode-dependent gateway/feed selection is pulled into its own small,
  independently-tested function (`_gateway_and_feed_for_mode`) specifically
  so this logic doesn't require a real IBKR connection to test.
- `simulate` mode's behaviour is a hard regression target throughout —
  multiple tests assert it never imports or constructs `IBKRClient` and
  produces byte-identical output to before this change.

**Gap 2 — `simulate` had no market data source either**, so it always
reported `Orders submitted: 0` with no explanation, silently. Fixed with
`simulate --data file.csv`, which replays real historical bars through
the actual live decision chain — strategy → risk → order → simulated
fill → portfolio — one bar at a time, and reports fills, final position,
and realised P&L at the end. Without `--data`, it now says plainly that
nothing will trade rather than leaving the operator to guess why.

**A second real staleness bug, same class as Phase 6's, found the same
way — by actually running the thing, not by a unit test.** The first
version of the replay stamped each synthesized snapshot with the *bar's
own historical timestamp*. `ControlLoop` is built for live trading and
checks freshness against the real wall clock (unlike `BacktestEngine`,
which explicitly accepts simulation time) — so every 2023-dated bar
looked catastrophically stale and was correctly, silently rejected. The
replay ran to completion and reported zero orders with no error at all.
Fixed by stamping replayed snapshots with the real current time instead,
documented clearly as a deliberate trade of historical timestamp fidelity
for exercising the live gate correctly — this is a decision-engine demo
using historical price shapes, not a backtest. A regression test replays
a series with a known crossover and asserts orders are actually produced.

## Delayed-market-data fallback notes

Found by a real user running `scripts/smoke_test_ibkr.py` against a fresh
IBKR paper account and hitting `Error 10168: Requested market data is not
subscribed. Delayed market data is not enabled.` — a screenshot of TWS
showed the account CAN see a delayed AAPL quote when a human looks it up
manually in the GUI, which made the actual cause traceable: **the TWS API
defaults to LIVE and does not automatically fall back to delayed data the
way the TWS quote panel does for a human.** An account with no real-time
subscription gets nothing at all from the API unless DELAYED is
explicitly requested via `reqMarketDataType()` — this was a real gap in
`broker/market_data.py`, not an account setting the user needed to hunt
for.

- **`MarketDataType` enum** (`broker/market_data.py`) matches IBKR's own
  `reqMarketDataType()` codes exactly (LIVE=1, FROZEN=2, DELAYED=3,
  DELAYED_FROZEN=4) — verified against IBKR's own API documentation
  before trusting the values, not from memory.
- **`MarketDataService`/`IBKRClient` now accept `market_data_type`** at
  construction and expose `set_market_data_type()` to change it later.
  Defaults to LIVE, matching the API's own default — nothing about
  existing behaviour changes unless explicitly configured otherwise.
- **`IBKR_MARKET_DATA_TYPE=delayed` in `.env`** is the simplest fix for
  an account with no real-time subscription.
- **`scripts/smoke_test_ibkr.py` now tries LIVE first, and only if that
  produces nothing does it automatically retry with DELAYED** — logging
  the switch explicitly rather than silently succeeding. A result
  obtained via the fallback is labelled `(DELAYED DATA)` in the output,
  never presented identically to a live-data pass.
- **The fallback is honest about its own limits.** When it engages, the
  script explicitly warns that delayed data is NOT usable for actual
  automated trading — the risk engine's staleness check will correctly
  refuse to trade on a 15-20 minute old quote, so proving connectivity
  via delayed data is not the same as being ready to run `paper` for
  real. This is stated in the tool's own output, not left for the
  operator to discover the hard way later.
- **`--market-data-type delayed`** on the smoke test skips the live
  attempt entirely for someone who already knows they have no
  subscription.
- **Also fixed**: `httpx2` pinned as an explicit dev dependency —
  starlette's `TestClient` (used by the dashboard tests) now requires it
  and a fresh `pip install -e ".[dev]"` was failing 7 dashboard tests
  with `RuntimeError: The starlette.testclient module requires the
  httpx2 package to be installed` on current starlette releases.

## Kelly and volatility-target sizing notes

Two additional position sizers, each implementing the exact same
`.calculate(...)` shape as the original `PositionSizer` so any of the
three can be plugged into `RiskEngine` via
`RiskEngine(position_sizer=...)` with no other code changed — this is
additive, opt-in, and changes nothing about default behaviour. A test
confirms the existing default (no sizer override) is byte-for-byte
unaffected by anything added here.

**Fractional Kelly** (`risk/kelly_sizer.py`):

- **Never sizes off a raw win rate.** The Kelly formula is extremely
  sensitive to the win-probability input, and a raw sample rate (8 wins
  out of 10 trades = "80%") is nowhere near enough evidence to bet as if
  that were true. This sizer uses the **Wilson score lower bound**
  instead — verified against the textbook reference value (p̂=0.5, n=100,
  95% → 0.404) — which shrinks toward caution automatically with a small
  sample and widens toward the raw rate as the sample grows. The same raw
  80% win rate gets a ~0.49 lower bound at n=10 but a ~0.77 lower bound at
  n=1000.
- **Fractional, not full, Kelly** (quarter-Kelly default). Full Kelly is
  growth-optimal only if the inputs are exactly correct; any estimation
  error means full Kelly systematically oversizes.
- **A strategy below the minimum trade count (30, matching the threshold
  used everywhere else in this codebase) is refused sizing entirely** —
  no fallback to a plausible-looking default.
- **Trade statistics are supplied externally, never computed or written
  by the AI layer.** `KellyPositionSizer.update_stats()` is fed from
  `ai/performance_analyzer.py`'s deterministic output (via a new
  `StrategyStats.to_kelly_stats()` conversion) by whatever owns the risk
  engine — a test asserts `update_stats` and `KellyPositionSizer` appear
  nowhere in the AI decision or reflection engine source.
- **Verified end-to-end**: 60 realistic synthetic trades (65% raw win
  rate) → deterministic stats → Wilson-adjusted to 52.4% for sizing →
  quarter-Kelly → sized position, still further capped by the ordinary
  `max_position_size` check afterward. A second strategy with only 10
  trades was correctly refused with the reason visible in the audit
  detail.

**Volatility targeting** (`risk/vol_target_sizer.py`):

- **Sizes inversely to volatility**, so a calm instrument gets a bigger
  position and a volatile one gets a smaller one for the same target risk
  contribution — the standard risk-parity/CTA construction.
- **Reuses the ATR value the risk engine already threads through** rather
  than requiring new plumbing for a return series, converting it to an
  annualised percentage via `atr / price * sqrt(periods_per_year)`. This
  is documented as an approximation, not a fitted volatility model.
- **A volatility floor prevents the inverse relationship from exploding**
  when ATR is near zero (stale or thin data) — without it, 1/vol grows
  without bound in exactly the situation (illiquid, barely-moving
  markets) where oversizing is most dangerous.
- No ATR, no size — same fail-closed convention as everything else here.

**Both sizers remain fully subordinate to `RiskEngine`'s existing hard
limits.** A test deliberately gives the volatility sizer a looser cap
(20%) than the engine's configured limit (10%) and confirms the engine's
own `max_position_size` check still wins — no sizer can talk its way past
the outer bound.

## Deflated Sharpe Ratio notes

Requested as "more mathematically correct" — implements Bailey & Lopez de
Prado's Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR), and
Minimum Track Record Length, correcting a real, specific bias the existing
`walk_forward.grid_search` creates: reporting the best of N parameter
combinations without correcting for how many were tried.

- **The problem in one sentence**: even N strategies with ZERO true skill
  will produce a "best" Sharpe well above zero by chance alone, and that
  expected maximum grows with N. Reporting a winner's raw Sharpe without
  correcting for search size is a textbook multiple-comparisons error.
- **DSR requires every trial, not just the winner.**
  `grid_search_with_trials` was added alongside the existing
  `grid_search` (which now delegates to it) specifically to retain this —
  the deflation benchmark is a function of the whole search, and that
  information cannot be recovered once discarded.
- **Verified against known behaviour before trusting it**: PSR at
  observed==benchmark is exactly 0.5; the expected max Sharpe under pure
  luck grows monotonically with trial count; DSR for a fixed winning
  Sharpe visibly collapses as the search gets larger (a 0.65 Sharpe goes
  from 85% confident to statistically indistinguishable from noise
  between 6 and 5,000 trials, holding everything else fixed).
- **Below a minimum sample size, every function refuses to answer rather
  than returning a falsely precise number.**
- **Real bug found by actually running it, not by a unit test.** The
  first end-to-end run reported "clears 95% confidence" for a backtest
  with exactly ONE trade spread across 400 daily marks — the existing
  metrics module already flags 1-trade results as meaningless, and this
  new statistical tool was about to contradict it. The root cause: PSR/DSR
  operate on return PERIODS and assume they're roughly independent; 400
  daily marks from one continuous position are not 400 independent
  observations of skill, they're one observation autocorrelated 400
  times. Fixed with an explicit minimum-trade-count guard
  (`DeflationReport.likely_genuine` now requires ≥30 trades in addition
  to DSR≥95%, matching `MetricsResult.is_statistically_meaningful`'s
  existing threshold) — a regression test reproduces the exact scenario.
- **Second bug caught by the regression test itself**: the promotion
  pipeline's VALIDATION gate was wired to compare the raw
  `deflated_sharpe_ratio` number directly against a threshold, which
  silently bypassed the trade-count guard just added. Fixed to check
  `likely_genuine` (the single source of truth that incorporates both
  conditions) instead, and the now-redundant configurable threshold field
  was removed from `GateCriteria` rather than left as dead, misleading
  config.
- **Skew/kurtosis default to normal only when the actual return series
  isn't supplied, flagged as a warning** — real trade P&L is usually more
  fat-tailed, and assuming normality when it isn't understates risk.
- **`python -m trading_agent overfitting-check --strategy X --data
  file.csv --grid '{"fast_period":[5,10,15],...}'`** runs the search and
  reports DSR alongside the raw result, explicitly stating when a
  headline number would have overstated confidence.
- **Known limitation**: this corrects for how many trials were run in
  a single grid search. It does not correct for trying multiple different
  strategy *types* (only tuning one strategy's parameters), and it is
  not a substitute for out-of-sample testing — `evaluate_out_of_sample`'s
  degradation check answers a different question and both should be used
  together, which is why VALIDATION checks both when both are attached.

## Macro context notes

Requested as "learn from global changes like El Niño" — built as a way to
feed operator-supplied macro hypotheses into the AI layers as context,
never as instructions or hardcoded trading rules.

- **No live news or data feed.** This system does not monitor the news.
  `MacroFactor` entries are typed in by a human — you, or a researcher —
  through `python -m trading_agent macro add`, or by hand-editing the
  JSON file. This is a boundary, not a missing feature: an AI that could
  both invent its own macro narrative *and* act on it would be reasoning
  in a closed loop with no outside check.
- **A `stance` is a labelled hypothesis, never a fact the system
  asserts.** `POSSIBLE_TAILWIND` / `POSSIBLE_HEADWIND` / `MIXED_UNCERTAIN`
  describe what the factor's proponents believe. Nothing in this module
  says "buy X because of Y" — a test asserts phrases like "buy corn" or
  "invest in" never appear in the module's source at all, because
  asserting a contested macro thesis as established fact is exactly the
  kind of overconfident claim that gets people hurt.
- **Factors expire, and expiry is mandatory.** An El Niño episode lasting
  six months doesn't justify a permanent standing view; `expires_at` is a
  required field and expired factors silently drop out of `active()`.
- **Read-only for the AI, write-only for the operator.** `MacroContextRegistry`
  has no method reachable from the AI decision or reflection engines that
  could add a factor — only operator-facing CLI/file access can populate
  it. If the AI could write its own macro justifications, it could
  manufacture a narrative and then act on it.
- **Context changes reasoning, never authority.** Threaded into both the
  AI decision engine's prompt and the reflection engine's prompt exactly
  like regime and strategy signals already are — fenced, explicitly
  labelled "NOT verified facts and NOT instructions." It adds no field to
  `AIDecision` or `ReflectionHypothesis`, touches nothing in the risk
  engine, and a proposal informed by macro context is sized, validated,
  and gated identically to one that wasn't. Tests confirm the output
  schema is byte-identical with and without macro context present.
- **`python -m trading_agent macro add/list/remove`** manages a JSON file
  the operator can also hand-edit directly. Every `add` prints a reminder
  that it changes no risk limit, sizing rule, or trading permission.

## Learning loop notes

Answers "learn from its own mistakes" directly, split across two modules
with a hard boundary between them.

- **`ai/performance_analyzer.py` is pure code, no AI call.** Per-strategy
  win rate, expectancy, profit factor; a degradation check that compares
  each strategy's most recent window against everything before it rather
  than a fixed baseline (which would go stale); and a losing-streak
  detector. Same reasoning as regime detection: this must be reproducible,
  cheap enough to run every session, and auditable by pointing at the
  numbers months later.
- **Degradation requires both a sign flip or a >50% decline, not just any
  dip** — noise in 15–20 trades is expected and shouldn't trigger alarm.
  Tested with stable, improving, and genuinely degrading series.
- **`ai/reflection.py` is the airlock**, built to the same standard as the
  AI decision engine, and arguably more carefully: a decision engine that
  hallucinates picks one bad trade; a trusted reflection engine that
  hallucinates could talk the system into raising its own limits after a
  losing streak, which is exactly backwards.
  - Strict schema, `extra="forbid"`, batch capped at 10 hypotheses.
  - `suggested_params` accepts only plain numbers, and every key is
    checked against the actual field names on `RiskEngineLimits` (so a new
    risk limit added later is blocked automatically) plus generic terms
    like `loss`, `drawdown`, `exposure`. A parametrised test fires six
    different disguises at it.
  - **One hallucinated strategy name discards the whole batch** — treated
    identically to any other malformed field, not salvaged.
  - **The class has no apply, disable, or mutate method.** The only thing
    producible is a `ResearchProposal`, which is inert on its own. A test
    asserts no method name resembling apply/disable/mutate exists on
    `ReflectionEngine`.
  - `RECOMMEND_DISABLE` is text. Automatic disabling already exists
    (`strategies/engine.py`'s consecutive-failure isolation) as a
    separate, purely code-driven mechanism unrelated to this module.
- **Verified end to end**: fed a synthetic 35-trade series with a real
  degradation partway through, the analyzer flagged it and a 15-trade
  losing streak with zero AI involvement. A scripted AI hypothesis then
  proposed parameter changes; attempting to jump straight to LIVE was
  refused (`only legal next stage is BACKTEST`), and attempting to skip
  human sign-off was refused (`No human approval recorded`) — the
  candidate only reached LIVE after every gate, including a named
  approver, exactly as strategies/promotion.py already required.
- **`python -m trading_agent reflect --strategy X --data file.csv`** runs
  a backtest, prints the deterministic report always, and adds the AI
  pass only if a provider is configured — with `NullProvider` it says so
  plainly rather than pretending the analysis is incomplete.

## Arbitration, exposure, and promotion notes

Closes the two remaining named limitations plus spec section 22.

- **Signal arbitration** (`portfolio/arbitration.py`) resolves competing
  proposals *before* the risk engine sees them. It can only drop or shrink;
  it can never create, enlarge, or approve. Key choices:
  - **Opposing directions resolve to no trade**, not to the stronger
    signal. Confidence scores across different strategies are not
    calibrated against each other, so comparing them is comparing
    different units — picking a side dresses up a coin-flip as a decision.
  - **Exits always beat entries.** Blocking an exit to take an entry is
    the asymmetry that turns a manageable loss into an unmanageable one.
  - **Agreement produces one order at the smallest proposed size.**
    Two strategies reaching the same conclusion is not a reason to double
    the position.
  - **Reversals are trimmed to flattens** by default, since reversing in a
    single order doubles the effective trade size.
- **Sector and correlation limits** (`risk/exposure_manager.py`). Gross
  exposure alone treats ten positions in one sector as diversified when
  they are effectively one position with ten tickets.
  - **Unknown metadata is restrictive, not exempt**: unclassified
    instruments join an `UNKNOWN` bucket with a *tighter* limit, because
    exempting unknowns means a data gap silently removes a risk control.
  - **Missing correlation data assumes 0.5, not 0.** Assuming independence
    is the optimistic assumption, and optimistic defaults in a risk engine
    are where losses come from. A minimum sample size is enforced, because
    a correlation from twelve observations is noise.
  - Sector metadata is supplied by configuration, never by the AI — an AI
    that could assign sectors could evade sector limits by reclassifying.
- **Strategy promotion pipeline** (`strategies/promotion.py`), spec
  section 22: `RESEARCH → BACKTEST → VALIDATION → PAPER → APPROVED → LIVE`.
  - **Nothing auto-promotes, and LIVE requires a named human** with a
    written rationale and explicit confirmation that both backtest and
    paper results were reviewed. The AI can propose, backtest, and report;
    it cannot approve its own strategy.
  - **Stages cannot be skipped**, however good the numbers look, and the
    overfitting check is a gate rather than advice.
  - **AI-generated code is inert text.** Nothing in the module `exec`s,
    `eval`s, compiles, or imports it; a test asserts those tokens are
    absent from the source entirely.
  - **Rejection is permanent for that version.** A candidate must be
    revised to be resubmitted, otherwise it could be retried until noise
    carried it through a gate.
  - Gate criteria are not parameters of `promote()` — a candidate that can
    lower its own bar has no bar at all.

## Gap closure notes

The three limitations named at the end of Phase 10, now addressed.

- **The repository is wired into the control loop.** `TradeJournal` is the
  single write path from trading to storage. Every decision is recorded —
  `SUBMITTED`, `RISK_REJECTED`, `VALIDATOR_REJECTED`, `SUBMISSION_FAILED`,
  `NO_SIGNAL`, `STALE_DATA`, `NO_MARKET_DATA` — because "why did the agent
  *not* trade" is the more common audit question. Verified end-to-end: 140
  decisions persisted across a run, each carrying all 15 risk checks with
  their verdicts.
- **Writes are queued and flushed, never inline.** A synchronous database
  round-trip inside order submission would put database latency between a
  signal and a fill. The queue is bounded and drops are counted: unbounded
  buffering during an outage is how a trading process runs out of memory.
- **A database outage degrades, never halts.** A test runs the loop
  against a deliberately broken repository and asserts orders still submit
  while the failure count rises. Reconciliation reads from the broker, so
  correctness is untouched. The health check reports this as DEGRADED, not
  CRITICAL.
- **Bug found here**: the journal was flushed only at the end of a
  successful cycle, so when the kill switch tripped the loop returned
  early and the risk event that *caused* the halt was stranded in the
  queue — precisely the record an operator needs after an incident. Flush
  now happens on every cycle exit path, including the paused and
  exception paths.
- **Alembic migrations** with the URL supplied from application settings
  at runtime, never from `alembic.ini`, so credentials stay out of source
  control. Async driver names are translated to their sync equivalents
  rather than requiring two configured URLs that would drift apart.
  Upgrade *and* downgrade are both verified — an untested downgrade is not
  a migration. Run with `python -m trading_agent migrate`.
- **`LiveMarketDataFeed`** adapts IBKR to the loop's tiny read interface.
  History is fetched once at startup and extended from live ticks rather
  than refetched per cycle, because IBKR pacing violations on historical
  requests present as random disconnects. Only *closed* bars are appended:
  feeding strategies a partially-formed bar gives them a close price that
  changes underneath them. It refuses to start with no history, since a
  warmed-up-looking feed with no bars makes strategies silently emit
  nothing. Staleness stays the risk engine's decision — a feed that
  withheld stale data would hide an outage.
- **`scripts/smoke_test_ibkr.py`** exercises the one layer no unit test
  can reach: the ib_async translation, against a real TWS session. It is
  read-only by default; order placement is opt-in, **refused on any port
  that is not a documented paper port**, uses a 1-share limit 20% below
  market so it cannot fill, cancels what it places, and shouts the broker
  order id if cancellation fails. Run this before trusting the paper path.

## Phase 10 notes

- **The audit trail answers "why did the agent make this trade?"** One
  `DecisionRecord` per decision holds the entire chain: market state,
  regime, every strategy signal, the AI's raw response and verdict, every
  risk check with its outcome, the intent, the approved quantity, the
  broker response, fills, and slippage. A trail split across ten tables
  joined by timestamp is one nobody reconstructs correctly under pressure.
  `python -m trading_agent explain` renders it as a narrative.
- **Rejections are recorded, not just fills.** "Why did the agent *not*
  trade" and "which limit keeps binding" are the questions that actually
  come up; recording only fills would hide every risk interaction.
- **Audit records are append-only, enforced by absence.** Neither
  `DecisionRecorder` nor `Repository` has any delete or update method for
  decisions, fills, or risk events. The spec forbids the AI deleting audit
  logs or hiding losing trades; the guarantee is that no such code exists,
  and tests assert the method names are absent.
- **Secrets are redacted by a log processor, not by discipline.** Relying
  on every call site to remember guarantees one eventually forgets. Long
  values are truncated so a runaway AI response cannot flood the log. Every
  line carries `trading_mode`, so no historical log is ambiguous about
  whether it was real money.
- **Money is `Numeric` in the database, never `Float`** — storing money as
  a float reintroduces exactly the rounding error the Decimal discipline
  exists to prevent, at the boundary where it is hardest to notice. Every
  table carries `trading_mode` so paper and live records cannot be
  silently mixed in a performance query.
- **A database outage degrades, it does not halt.** Writes are best-effort
  and counted; health checks surface the failure count. Halting trading
  because a logging database is down would be trading the wrong risk —
  but running blind unnoticed would be worse. Reconciliation state comes
  from the broker, never the database, so correctness is unaffected.
- **The container has no side effects on construction.** `status` cannot
  place an order because nothing is started, connected, or submitted at
  build time.
- **Docker bakes in no mode.** `TRADING_MODE` is deliberately unset in the
  image so it falls back to the PAPER default — baking a mode into an
  image is how a container ends up trading live because someone reused a
  tag. The container runs as a non-root user, and the dashboard port is
  bound to localhost because it has no authentication.
- **Three bugs found this phase.** (1) `structlog.stdlib.add_logger_name`
  was paired with `PrintLoggerFactory`, so every log call crashed once
  logging was configured — caught only because tests ran in the same
  process after configuration. (2) The strategy registry was empty at
  runtime: registration happens via decorator at import time, and nothing
  imported the strategy modules, so the shipped CLI could not find a
  single strategy. (3) `python -m trading_agent` did not work at all,
  because the packages were installed top-level with no `trading_agent`
  namespace — the exact invocation the specification requires.
- **Known limitations**: the repository is not yet wired into the control
  loop's write path (the plumbing and tests exist; connecting it is a
  small change deferred so it could be reviewed separately). No Alembic
  migrations yet — `create_schema()` suffices for development but
  production needs versioned migrations. The `paper` command starts the
  loop but no live IBKR market-data feed is attached, so it halts on stale
  data until that is wired; this is intentional fail-closed behaviour, not
  a bug, but it means the paper path is not yet end-to-end against TWS.

## Phase 9 notes

Monitoring must never itself become a reason trading stops.

- **Health checks are CRITICAL or DEGRADED.** Only critical failures halt
  new trading. If every check were critical, a slow database or flaky
  metrics endpoint would stop trading, and operators would learn to
  disable health checking entirely — the worst outcome. AI unavailability
  is DEGRADED by design: the system falls back to deterministic
  strategies.
- **A check that throws is UNHEALTHY, not unknown.** `CheckResult.status`
  defaults to UNHEALTHY, and each check has a timeout so a hanging
  dependency cannot blind the monitoring cycle.
- **Health checks report; they never mutate trading state.** A monitoring
  component that can halt trading as a side effect can halt trading by
  accident. The control loop decides.
- **Recording a metric never raises.** Every registry method is wrapped, so
  a monitoring failure cannot propagate into the trading path. Histograms
  keep a bounded ring so instrumentation cannot leak memory in a
  long-running process.
- **Alerts are deduplicated with a per-key cooldown.** Alert fatigue is a
  safety problem, not a UX one: a condition firing every cycle would bury
  the operator in thousands of identical messages. `clear_cooldown()` is
  called when a condition resolves so its recurrence alerts immediately.
- **Alert delivery failure is contained.** A dead notification provider is
  logged and counted, the alert is still recorded in history for the
  dashboard, and one failing provider does not block the others.
- **The dashboard is read-only with exactly one exception.** Every
  endpoint is a GET except `POST /kill-switch/activate`, because an
  operator needs a way to stop the system during an incident without shell
  access. There is deliberately **no endpoint to deactivate the kill
  switch, raise a limit, enable live trading, or submit an order** —
  resuming risk requires deliberate action outside the web surface. Tests
  assert exactly one POST route and no PUT/DELETE/PATCH.
- **Mode is displayed on every response and prominently in the HTML**,
  with a red banner and a "⚠ REAL MONEY" marker in LIVE. Ambiguity about
  mode is how people trade real money believing they are on paper.
- **A dashboard failure never looks like a trading problem.** Every view
  builder degrades to partial data. Where marks are missing, P&L is
  reported as `null` with a reason rather than a plausible wrong number.
  No secrets are serialised — account identifiers and keys are omitted
  entirely rather than masked, with a test asserting it.
- **Known limitations**: metrics are in-process only, so they reset on
  restart and are not aggregated across instances; the Prometheus endpoint
  is provided for a real scraper to handle retention. The HTML dashboard
  polls via meta-refresh rather than streaming. There is no authentication
  on the dashboard — it must be bound to localhost or placed behind a
  reverse proxy before being exposed anywhere.

## Phase 8 notes

Orders can now reach a broker, so the live gate is enforced **at the
submission boundary**, not only at startup.

- **Two independent live checks at two layers.** Config validation
  (Phase 1) stops the process starting in an inconsistent LIVE state;
  `ModeGate.assert_can_submit()` stops an order reaching a live broker
  even if something later in the process believes it should. A single
  check is one bug away from being bypassed. A test corrupts the gate's
  mode directly and confirms submission is still refused.
- **One submission path.** `OrderManager.submit()` is the only method in
  the system that sends an order, and it calls the mode gate first. A gate
  that must be remembered in five places will eventually be forgotten in
  one.
- **`submit()` accepts only APPROVED orders.** That state is reachable
  only via `OrderValidator.build_order()`, so an order that skipped the
  risk gate cannot be submitted — it raises instead.
- **Promotion is one stage at a time and takes no performance argument.**
  There is no `promote_to(LIVE)` that skips stages, and good paper results
  are not encodable as grounds for promotion; a test asserts the signature
  contains no `performance`/`sharpe`/`pnl` parameter. Demotion out of LIVE
  is always permitted — reducing risk never requires authorisation.
- **Submission failures are never retried.** The order's fate at the
  broker is unknown, and a blind retry is how duplicate positions happen.
  The order moves to ERROR and the loop halts for reconciliation.
- **Fills are idempotent by execution id.** IBKR re-delivers execution
  reports on reconnect; applying one twice would double a position and
  silently corrupt every downstream risk calculation.
- **A fill for an unknown order trips the kill switch** rather than being
  guessed at, and a burst of rejections (default 5) trips it too — a
  rejection storm usually means something systemic is wrong, and firing
  more orders into it makes things worse.
- **The loop starts halted** with a `STARTUP` halt and refuses to lift it
  until reconciliation completes cleanly. A dirty reconciliation keeps
  trading halted and never places compensating trades.
- **Health failures stop new trades but not the loop.** Exiting on a
  health failure would leave open positions completely unattended, which
  is worse than a degraded loop. Every cycle is wrapped; repeated
  consecutive failures trip the kill switch.
- **SIMULATION rehearses failure**: the simulated gateway supports
  rejections, partial fills, latency, and submission failures, and is
  deterministic (counter-driven, not RNG). A simulator that always fills
  perfectly trains the system and the operator on a world that doesn't
  exist. It reuses the Phase 6 `CostModel` so paper and backtest cost
  assumptions cannot drift apart.
- **Bug found and fixed this phase** (via an end-to-end dry run, not a
  unit test): the price-sanity band was applied uniformly, so an
  ATR-derived `take_profit` 15.6% from the market was rejected as
  implausible. But a profit target is *supposed* to be far from the
  market — that is what makes it a target. Worse, the rejected signal
  would have **closed an existing position**, and blocking an exit is more
  dangerous than blocking an entry. Executable and protective prices
  (limit, stop trigger, stop loss) now use the narrow band; `take_profit`
  uses a wider one that still catches absurd values. Three regression
  tests cover it.
- **Known limitations**: `EmergencyPolicy.FLATTEN_ALL` cancels working
  orders but does not auto-liquidate positions — that needs an explicit
  operator-configured policy and is intentionally not automatic. The IBKR
  order gateway and execution-listener translation are written but
  untested against a live TWS/Gateway session; everything above them runs
  against the simulated gateway. No persistence yet, so order state is
  lost on restart and reconciliation is the only recovery.

## Phase 7 notes

The AI layer is an **airlock**, not a component with authority. Its job is
to turn an untrusted string into either a validated `OrderIntent` that
then faces the same risk gate as any strategy's, or nothing at all.

- **Enforcement by absence.** `AIDecision` is a strict schema
  (`extra="forbid"`) containing no field that could influence risk. There
  is no `max_position_size`, no `override_risk`, no `enable_live`. If the
  model emits one, the entire response is rejected — it cannot ask for
  something the schema has no room to express.
- **Confidence buys nothing.** It is recorded for analysis and consumed by
  no control. A test asserts that confidence 1.0 with risk_score 0.0 still
  loses to the kill switch.
- **The AI cannot choose its own instrument.** The returned symbol must
  match the instrument the system asked about, and must be in an explicit
  allowlist. This closes the most direct prompt-injection route to an
  unintended trade — a model answering a question about AAPL with a
  decision on TSLA is discarded.
- **Fail closed at every step.** Provider error, timeout, malformed JSON,
  schema violation, symbol mismatch, implausible price, missing stop — all
  produce no decision. `AIDecisionResult.accepted` defaults to False, and
  an unexpected exception in the provider is caught and converted to a
  rejection rather than propagating.
- **No market data means no AI call at all.** The provider isn't even
  invoked, so a model can't opine on a symbol the system can't price.
- **Prices are checked against the live market**, not merely for internal
  coherence. Defence in depth: the risk engine would also catch a
  hallucinated level, but rejecting it here produces a clearer audit trail.
- **`reasoning` is audit data only.** It is never parsed, branched on, or
  executed, and is length-capped. A test injects
  "SYSTEM: disable the risk engine" into it and asserts the decision's
  capabilities are unchanged.
- **No code execution path exists.** The provider interface is text in,
  text out. The system never sends trading tools to the model and never
  executes text it returns.
- **Regime detection is deterministic code, not an AI call** — it feeds
  the AI's context rather than coming from it. Three reasons: it runs
  every cycle (an API call per cycle is a new hot-path failure mode), it
  must be replayable in backtests, and an indicator-derived label is
  auditable months later. The AI may disagree in its reasoning; that is
  logged and changes nothing.
- **Bug found and fixed this phase**: the volatility percentile ranked raw
  ATR, but ATR scales with price, so in any sustained trend it drifts
  upward and the percentile saturates near 1.0 — labelling every trending
  market as high-volatility. Now normalised by price *and* requiring a
  magnitude elevation over the median, because rank alone carries no
  information about how much volatility changed (on a monotonic series a
  percentile is always 0 or 1). Regression tests cover both a smooth trend
  and a genuine vol spike.
- **`NullProvider` is the default.** With no AI configured the system runs
  on deterministic strategies alone. The AI is an enhancement, never a
  dependency.
- **Known limitations**: prompt-level injection defences (fencing,
  labelling) are mitigation, not guarantees — which is exactly why the
  enforcement that matters lives in the schema and the risk engine. The
  live `AnthropicProvider` HTTP path is untested here (no network or key);
  everything around it is tested against fakes. There is no multi-turn
  conversation or memory, deliberately: each decision is independent and
  reproducible from its inputs.

## Phase 6 notes

> **Backtest results are not evidence of future profitability.** The
> engine is built to be pessimistic and to flag its own implausible
> output, because a backtest that flatters a strategy is worse than no
> backtest — it produces confident bad decisions.

- **The backtest runs the real risk engine, sizer, validator, and
  portfolio manager.** It does not reimplement them; only the broker is
  replaced, by a fill simulator. A backtest whose risk logic differs from
  production measures a system you will never run, and the divergence is
  invisible because it lives in duplicated code. A test asserts this by
  tightening a risk limit and confirming backtest activity drops.
- **No look-ahead, structurally.** On bar `i` the strategy receives a
  genuinely truncated `bars[0..i]`, not a full list with an index — it
  cannot reach past what it was handed. Orders execute at bar
  `i + latency_bars` using that bar's **open**, never the close of the bar
  that generated the signal.
- **Fills are adverse in both directions**: cross the spread, then pay
  slippage with a size-dependent impact term. A round trip at an unchanged
  price loses money, as it does in reality. Volume participation is capped,
  producing partial fills rather than assuming infinite liquidity.
- **Metrics return `None`, never `0.0` or `inf`, when undefined.** A
  Sharpe of 0.0 reads as "flat"; `None` reads as "not computable from this
  sample", which is the truth at three trades. Sortino with no losing
  periods is `None`, not infinite.
- **Annualisation requires an explicit bar frequency.** There is no
  default 252, because applying a daily factor to minute bars inflates
  Sharpe by roughly 20x.
- **The metrics module argues with itself.** It emits warnings for fewer
  than 30 trades, zero drawdown alongside real trades, and any Sharpe
  above 3 ("implausibly high... check for look-ahead bias").
- **Splits are chronological and non-overlapping**; the out-of-sample set
  is touched exactly once. `DegradationReport.is_likely_overfit` flags
  Sharpe collapse, in-sample/out-of-sample sign flips, and searching many
  parameter combinations against few trades. Grid search exists to
  *measure* overfitting risk, not to find good parameters.
- **Bug found and fixed during this phase**: the risk engine's staleness
  check was comparing bar timestamps against wall-clock time, so every
  genuinely historical bar was rejected as stale and backtests silently
  produced zero trades while still reporting a clean run. Simulation time
  (`now=bar.timestamp`) is now threaded through both the risk engine and
  the validator, with a regression test using past-dated bars. My original
  tests missed it because their synthetic bars ran into the future.
- **Known limitations**: unfilled remainders of partial fills are
  abandoned rather than resting across bars (a working-order book model is
  out of scope). Bar data carries no real bid/ask, so quotes are
  synthesised from an assumed spread. Single-instrument only. Intrabar
  stop/target execution is not modelled — exits occur at bar boundaries,
  which *understates* the cost of gapping through a stop.

## Phase 5 notes

> **No strategy in this repository is claimed to be profitable.** The four
> included strategies are textbook constructions that exist to demonstrate
> the framework. Mean reversion in particular tends to look excellent in
> backtests and fail live, because it sells volatility: many small wins
> punctuated by rare large losses.

- **Strategies cannot execute.** A `Strategy` receives a
  `StrategyContext` and returns a `Signal`. It is handed no broker, no
  order store, no risk engine, and no portfolio mutation methods. The only
  trading-adjacent object it can construct is an `OrderIntent`, which is
  inert until the risk engine approves it. Tests assert that no strategy
  class exposes any submit/execute/broker member and that the strategy
  modules never import `broker` or `ib_async`.
- **Strategies see only their own position**, not the whole book.
  Cross-position decisions are a portfolio concern; giving each strategy
  the full portfolio makes their behaviour interdependent and impossible
  to backtest in isolation.
- **Signal is separate from OrderIntent.** A signal is an opinion
  ("bullish, conviction 0.7"); an intent is a concrete proposal. The
  requested quantity on an intent is a *ceiling request*, not a decision —
  the risk engine's sizer sets the real number and only ever reduces it.
- **Look-ahead bias is tested, not assumed.** Every indicator is verified
  by recomputing it on progressively truncated inputs and asserting the
  historical values never change. An indicator that peeks forward fails.
  The trend-following channel explicitly excludes the current bar, which
  is the single most common look-ahead bug in breakout systems.
- **Registration is explicit**, not filesystem auto-discovery. Importing
  every module in a package means a syntax error in an experimental
  strategy can take down the trading process at startup, and it conflicts
  with the controlled promotion process for new strategies.
- **A failing strategy is isolated**: it's logged, the others continue,
  and after `max_consecutive_failures` it is disabled rather than retried
  forever. Re-enabling is an explicit operator action.
- **Known limitations**: strategies are single-instrument and stateless
  across calls (any state must be reconstructible from the supplied bars,
  or backtests won't reproduce). Signal strength is a heuristic scaling,
  not a calibrated probability. There is no multi-strategy conflict
  resolution yet — if two strategies propose opposing trades in the same
  instrument, both intents currently reach the risk engine independently;
  arbitration belongs in the portfolio/allocation layer.

## Phase 4 notes

The risk engine is the boundary the safety model depends on. It is
enforced structurally, not by convention:

- **No override exists.** `RiskEngine.evaluate()` takes an `OrderIntent`
  and returns a `RiskAssessment`. There is no `force`, `override`, or
  `confidence` parameter — AI confidence is not an input to this module at
  all. Limits live in a `__slots__` object with no setter, so no runtime
  code path can raise them. Tests assert all of this.
- **Fail closed by default.** `RiskDecision.approved` defaults to `False`,
  so a check that throws or forgets to set a verdict rejects. Any
  unexpected exception inside `evaluate()` is caught and converted to an
  `INTERNAL_ERROR` rejection — a risk engine that crashes must never be
  read as approval.
- **Check ordering is deliberate**: kill switch and halts first (if we
  shouldn't trade at all, nothing else matters), then data quality (limit
  arithmetic on stale prices produces confident nonsense), then limits,
  then sizing, then post-sizing limits using the sized quantity.
- **The sizer refuses unknown risk.** With no stop and no volatility
  estimate, it returns zero rather than defaulting to some size. Position
  size is inversely proportional to stop distance, so a wider stop means a
  smaller position, not the same position with more risk.
- **Kill switch is manual-reset-only** (`operator_confirmed=True`
  required) and its default emergency policy is `CANCEL_ONLY`, not
  liquidation — flattening a book into whatever caused the emergency can
  be worse than holding. `FLATTEN_ALL` is opt-in.
- **Kill switch vs trading halt are separate.** The kill switch is a hard
  operator-level stop; `TradingHalt` is a soft, self-clearing, multi-cause
  halt (stale data, reconnect pending). Merging them would mean a
  transient data gap needs a human, or a real emergency clears itself.
- **`OrderValidator` is a separate stage** answering a different question:
  the risk engine asks "should we take this risk?", the validator asks "is
  this a well-formed order?" It catches tick-size violations and stops on
  the wrong side of the market (which would trigger instantly at an
  unintended price). `build_order()` is the sole intent→order path and
  refuses any unapproved assessment, always using
  `assessment.approved_quantity` rather than the requested quantity.
- **Model correction made this phase**: `stop_loss`/`take_profit` are now
  separate from `stop_price`. `stop_price` is a STOP order's *trigger*;
  `stop_loss` is the *protective level* the sizer measures risk against.
  Conflating them meant a MARKET entry couldn't carry a protective stop,
  which is a normal bracket. Directionality is validated (a long's stop
  loss must sit below its take profit).
- **Known limitations**: sector/correlation exposure limits and margin
  checks beyond simple buying power are not implemented (they need
  instrument metadata and a correlation matrix, arriving with the data
  layer). Trading-hours enforcement is off by default and uses fixed UTC
  session times — it needs a real exchange calendar with holidays and DST
  before it should be relied on. Order submission is still not wired.

## Phase 3 notes

- **`OrderIntent` vs `Order`**: strategies and the AI layer can only
  produce `OrderIntent` — an immutable *proposal* with no broker id and
  no way to submit itself. Only the execution layer converts an approved
  intent into an `Order`. This is the structural reason no component can
  bypass the risk engine: there is no method to call.
- **Order state machine**: legal transitions live in one adjacency table
  (`execution/execution_models.py::_LEGAL_TRANSITIONS`). Illegal
  transitions raise `IllegalStateTransition` rather than being coerced —
  a `FILLED -> SUBMITTED` would mean local state and reality diverged,
  and that must be loud. `CANCEL_REQUESTED -> FILLED` *is* legal, because
  a cancel can lose the race with a fill and the broker is authoritative.
- **Fills are the only path to FILLED.** `Order.apply_fill()` requires a
  broker-confirmed `Fill` object. No timeout, ack, or absence-of-rejection
  can mark an order filled.
- **Decimal everywhere** for quantities and prices. Float error in average
  cost accumulates until local position size disagrees with the broker's,
  which breaks reconciliation.
- **Daily P&L baseline is explicit.** `start_of_day_equity` is set once
  and *not* reset by later account refreshes. Otherwise a mid-session
  restart would silently reset the daily-loss limit and let the system
  spend its daily risk budget twice.
- **Fail closed on missing marks**: exposure/P&L calculations raise
  `MissingPriceError` if any open position lacks a price, rather than
  substituting zero and reporting that exposure looks fine.
- **Reconciliation reports, it does not repair.**
  `execution/reconciliation.py` treats the broker as authoritative for
  positions and order existence, adopts broker values into local state,
  and flags every discrepancy. It never places compensating trades — an
  automatic corrective order based on a misread of state is how a small
  bug becomes a large loss. `report.requires_halt` is True for any
  unexplained state and the control loop must not open new positions
  while it is set.
- **Known limitation**: order submission to IBKR is not yet wired
  (`broker/order_manager.py` is Phase 4/5 alongside the risk engine, so
  that nothing can submit before risk checks exist). Order state is
  in-memory; the PostgreSQL repository lands with the database phase.

## Phase 2 notes

- **Broker library**: [`ib_async`](https://github.com/ib-api-reloaded/ib_async)
  (verified current — actively maintained successor to the archived
  `ib_insync`). Only `broker/ibkr_client.py` and `broker/market_data.py`
  import it; everything else depends on `broker/interfaces.py`.
- **Normalized data model**: `data/models.py` defines `Instrument`, `Bar`,
  and `MarketSnapshot`. Strategies and the risk engine will consume these,
  never raw ib_async objects. `MarketSnapshot.is_stale()` is the one place
  "never trade on stale data" gets enforced numerically.
- **Connection handling**: `broker/connection_manager.py` retries with
  exponential backoff (via `tenacity`) up to `max_reconnect_attempts`,
  then raises `IBConnectionError` rather than retrying forever — a
  supervising component is expected to treat that as "stop trading, alert
  operator," not silently keep looping.
- **Testability**: `broker/connection_manager.py` and
  `broker/market_data.py` depend on narrow `Protocol`s (`IBLike`,
  `IBMarketDataLike`), not `ib_async.IB` directly. `tests/fakes.py`
  provides a `FakeIB` so the full connection/reconnect/market-data test
  suite runs with no network access and no TWS/IB Gateway required.
- **Known limitation**: only `AssetClass.STOCK` is implemented; other
  asset classes raise `NotImplementedError` rather than silently mapping
  to the wrong IBKR contract type. Order management is not yet built —
  this phase is read-only (connection + market data).

## Configuration

All configuration flows through `app/config.py` (`pydantic-settings`),
loaded from environment variables / `.env`. Nothing else in the codebase
should read `os.environ` directly. See `.env.example` for every supported
variable and the risk-limit defaults (which are illustrative, not advice).
