# Complete Beginner's Guide: Running the Trading Agent on a Mac

This guide assumes you have **never used a computer's Terminal before**. Every single step is spelled out — including how to check that the step actually worked before moving to the next one. That last part matters: most problems people hit come from a step silently not doing what they expected, then trying the *next* step in the wrong place. This guide stops you at every point where that can happen.

Go slowly. Copy and paste commands rather than typing them by hand wherever you can — a single wrong character is the difference between a command that works and one that doesn't, and copy-paste removes that risk entirely.

---

## PART 0: READ THIS FIRST — Money and Risk

Before anything technical, understand what this software is.

This is a **trading system**. Its purpose is to connect to a stockbroker (Interactive Brokers) and buy and sell things automatically, without a human clicking the buttons.

Four things to understand before you continue:

**1. Trading loses money for most people who try it.** Automating it doesn't change that — it only makes whatever happens (winning or losing) happen faster and without you watching every second.

**2. Nothing in this system is claimed to be profitable.** The strategies included are demonstrations of how the software is built, not proven money-makers. Treat every result as unproven until you've personally tested it for months.

**3. A "backtest" is not a prediction.** It replays old price history and asks "what would this have done?" It's very easy to build something that looks brilliant on old data and loses money on new data. This software actively warns you when results look too good to be true — take those warnings seriously; they exist because the statistics are built to catch exactly that trap, and they've already caught it during testing.

**4. This guide never turns on real-money trading.** Everything below runs in **PAPER mode** — pretend money in a practice account. Turning on real trading requires deliberate steps this guide doesn't cover, on purpose.

If you're not fully comfortable with the possibility of losing money, using this purely as a learning tool in paper mode — forever — is a completely legitimate choice.

---

## PART 1: Understanding the Basics

### 1.1 What is the Terminal?

Most of the time you use a Mac by clicking things. The **Terminal** is a different way in: you type instructions instead of clicking. Clicking is like pointing at what you want in a shop; the Terminal is like writing the shopkeeper a precise note.

### 1.2 Opening the Terminal

1. Hold **Command** (next to the spacebar) and press the **spacebar**. Let go.
2. Type: `terminal`
3. Press **Enter** (also called Return).

A mostly-blank window opens with a blinking cursor after some text. That text is called the **prompt**, and it looks something like:

```
davidkoktavy@Davids-MacBook ~ %
```

**The `~` in that prompt matters — it tells you which folder you're standing in.** More on that in a moment; it's the thing that caused earlier errors, so we're going to get it right from the start this time.

### 1.3 Making sure you're in zsh, not bash

You may have seen a message like:

```
The default interactive shell is now zsh.
```

That's Apple telling you something true and slightly confusing: modern macOS uses a shell program called **zsh** by default, but your Mac still also has an old one called **bash** installed (frozen at a very old version, 3.2, for licensing reasons). If you ever see a prompt ending in a `$` that looks like `bash-3.2$`, **you are actually inside bash, not zsh** — and that's worth fixing before we go further, because every command and prompt shown in this guide assumes zsh, which is what a fresh Terminal window gives you by default.

**Check which one you're actually in:**

```bash
echo $SHELL
```

It should print:

```
/bin/zsh
```

If it instead prints `/bin/bash`, fix it:

```bash
chsh -s /bin/zsh
```

Enter your Mac password when asked (nothing shows on screen as you type it — normal). Then **completely quit Terminal and reopen a fresh window** (Command+Q to quit, then reopen).

**A quick visual way to tell them apart going forward, without typing anything:** zsh's default prompt ends in a percent sign `%`; bash's ends in a dollar sign `$`. Every prompt example in this guide (like `davidkoktavy@Davids-MacBook ~ %`) ends in `%` — if yours ever ends in `$` instead, you've somehow ended up in bash again, and running `echo $SHELL` is the way to confirm it.

None of the actual commands in this guide behave differently between the two shells — `cd`, `ls`, `pwd`, `source`, all work identically either way. The reason to standardize on zsh is simpler: it's what Apple ships as the real default now, it's what this guide's prompts and troubleshooting assume, and mixing the two across different Terminal windows is exactly the kind of quiet inconsistency that causes a step to "not work" for no obvious reason.

### 1.4 Typing a command

Boxes like this in this guide mean "type this exactly, then press Enter":

```bash
echo hello
```

Try it. You should see `hello` printed back. That's `echo` — it repeats what you tell it.

**Rules that matter:**

- **Copy-paste beats typing.** Select text with your mouse, **Command+C** to copy, click inside the Terminal window, **Command+V** to paste, then Enter.
- **Spaces and capital letters matter.** `cd Desktop` and `cd desktop` can be treated as different things.
- **After every command that's supposed to move you somewhere or set something up, we will CHECK that it worked before continuing.** This is the single biggest change from a normal tutorial, and it's there because of errors that have already happened once — every one of them was a case where a step silently didn't do what was expected, and the next command was then run in the wrong place.

**To stop a stuck command:** hold **Control** (not Command) and press **C**.

### 1.5 Folders, paths, and `~` — get this right and almost everything else works

Your Mac organizes files into **folders** (also called directories), nested inside each other like a filing cabinet.

**`~` is a shortcut that means "my home folder"** — for you, that's exactly `/Users/davidkoktavy`. Nothing more, nothing less.

Here is the exact mistake that caused an earlier error: you can't put `~` *and* `/Users/davidkoktavy` in the same path — that means "my home folder, followed by ANOTHER copy of my home folder," which doesn't exist. So:

| Written as | Means | Valid? |
|---|---|---|
| `~/Downloads` | `/Users/davidkoktavy/Downloads` | correct |
| `/Users/davidkoktavy/Downloads` | `/Users/davidkoktavy/Downloads` | also correct — same place, no `~` needed |
| `~/Users/davidkoktavy/Downloads` | `/Users/davidkoktavy/Users/davidkoktavy/Downloads` | doubled, does not exist |

**Rule: use `~` on its own, OR the full `/Users/davidkoktavy/...` path — never both together.** Every command in this guide uses `~` consistently so you never have to think about this again, but it's worth understanding why, because you'll eventually type a path from memory and this is the rule that keeps it correct.

**The core commands for moving around:**

```bash
pwd
```
"Print working directory" — tells you exactly where you are right now. **Use this constantly. When in doubt, run `pwd`.**

```bash
ls
```
Lists what's inside the folder you're currently in.

```bash
cd Desktop
```
Moves into a folder named `Desktop` *that must already exist inside your current folder*. If it doesn't exist there, you'll get "No such file or directory" — that error means "this folder isn't here," not "something is broken."

```bash
cd ..
```
Moves up one level (out of the current folder, into its parent).

```bash
cd ~
```
Jumps straight to your home folder from anywhere.

**Try this now, as practice, typing each line and pressing Enter, checking the result before moving to the next:**

```bash
cd ~
pwd
```
That `pwd` should print exactly `/Users/davidkoktavy`. If it doesn't, stop and re-read this section before continuing.

---

## PART 2: Installing Python

The software needs **Python 3.12 or higher**. If you have Xcode but not Homebrew, we'll install Python the simplest way that doesn't need Homebrew: the **official installer from python.org**.

### 2.1 Check what you already have

```bash
python3 --version
```

- **3.12 or higher** printed -> skip to Part 3.
- **Lower, or an error** -> continue below.

### 2.2 Install from python.org

1. In your browser, go to **python.org/downloads/macos**
2. Click the yellow download button for the latest **3.12.x** (or newer) macOS installer. It downloads a file ending in `.pkg`.
3. Open your **Downloads** folder in Finder, double-click that `.pkg` file.
4. Click **Continue** through the introduction, agree to the licence, click **Install**.
5. Enter your **Mac password** when asked. Nothing visibly appears as you type it — that's normal, just type it and press Enter/Return or click the button.
6. Wait for "The installation was successful," then **Close**.

### 2.3 Verify it — and this time we check properly before moving on

**Close the Terminal window completely and open a brand new one** (Command+Space, `terminal`, Enter). This step is easy to skip and is the single most common reason the next command fails — a Terminal window opened *before* you installed Python won't know it exists yet.

```bash
python3.12 --version
```

You should see `Python 3.12.x`. **If you see "command not found":**

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 --version
```

If *that* prints a version number, remember this exact long path — we'll use it once in Part 4, and never again after that.

Do not proceed to Part 3 until one of those two commands successfully printed a Python version number.

---

## PART 3: Getting the Project Onto Your Mac — carefully, this time

This is exactly where the earlier errors happened, so we're doing it in a way that makes the mistake structurally impossible.

### 3.1 Find the file, exactly

```bash
ls ~/Downloads
```

Look at the output. Find the file you downloaded — it will be named something like `trading_agent_complete.zip`. **Copy that exact name from what's printed on your screen.** Don't retype it from memory; that's how names drift and paths break.

If you don't see it there, the download may have gone to a different folder — check your browser's download history to see where it saved.

### 3.2 Make a clean, clearly-named home for the project

```bash
cd ~
mkdir -p Projects
cd Projects
pwd
```

That `pwd` must print exactly `/Users/davidkoktavy/Projects`. If it doesn't, stop here and figure out why before continuing — every later command in this guide assumes you're building from this exact folder.

### 3.3 Unzip it — using a name that cannot collide with anything inside it

Here's the specific trap from before: this project's internal file structure happens to contain a small subfolder that is *also* named `trading_agent` (it's just two tiny files that make one specific command work later — nothing important lives in it). If you *also* name the folder you extract everything into `trading_agent`, you end up with two different folders with the same name, nested inside each other, and it's very easy to `cd` into the wrong one.

To make this impossible, we'll give the extraction folder a **different, unmistakable name**: `tradingbot`.

```bash
unzip ~/Downloads/trading_agent_complete.zip -d tradingbot
```

(If the filename you saw in step 3.1 was different, use that exact name instead of `trading_agent_complete.zip`.)

You'll see a long list of files scroll past as it unzips. That's normal.

### 3.4 Verify you're looking at the real project root — do not skip this

```bash
cd tradingbot
pwd
```

That `pwd` must print exactly `/Users/davidkoktavy/Projects/tradingbot`.

Now the important check:

```bash
ls pyproject.toml
```

**This must print `pyproject.toml` with no error.** This file only exists at the true root of the project. If you get "No such file or directory" here, you are in the wrong folder — most likely the unzip created one extra layer of nesting. In that case run:

```bash
ls
```

and look for a folder inside that might contain the real project (sometimes a zip contains one top-level folder wrapping everything). If you see one, `cd` into it and run `ls pyproject.toml` again.

**Do not proceed past this point until `ls pyproject.toml` succeeds.** Every command for the rest of this guide assumes you're standing in this exact folder.

### 3.5 The one folder to never `cd` into by mistake

While you're here, look at what's inside:

```bash
ls
```

You'll see folders like `app`, `broker`, `risk`, `strategies`, `tests` — and also one literally named `trading_agent`. **That inner `trading_agent` folder is not the project.** It contains exactly two small files that exist only so a command later in this guide (`python -m trading_agent ...`) works. You never need to `cd` into it, ever. If you ever run a command and get an error mentioning "does not appear to be a Python project," the first thing to check is `pwd` — you've likely wandered into that inner folder by accident, and the fix is `cd ..` to step back out.

> **From now on, every command in this guide assumes you're in `~/Projects/tradingbot`.** If you close the Terminal and come back later — even the next day — the very first thing you do is:
> ```bash
> cd ~/Projects/tradingbot
> pwd
> ```
> and confirm it prints `/Users/davidkoktavy/Projects/tradingbot` before typing anything else.

---

## PART 4: Setting Up the Software

### 4.1 What a "virtual environment" is, briefly

Python programs need extra add-on packages, and different projects can need different, conflicting versions of the same package. A **virtual environment** is a private, isolated box for one project's packages, so nothing on your Mac interferes with anything else.

You create it once. You **activate** it every time you sit down to work on the project — every single session, not just the first time.

### 4.2 Create it

Confirm you're in the right place first — this habit will save you every time from here on:

```bash
pwd
```

Should print `/Users/davidkoktavy/Projects/tradingbot`. If it does, continue:

```bash
python3.12 -m venv .venv
```

(If `python3.12` wasn't found back in Part 2 and you had to use the long path instead, use that same long path here in place of `python3.12`.)

This takes a few seconds and prints nothing — that's normal, not a sign it failed.

### 4.3 Verify it was actually created before trying to activate it

This is the check that would have caught the earlier "Permission denied" error immediately:

```bash
ls .venv/bin/activate
```

**This must print `.venv/bin/activate` with no error.** If you get "No such file or directory," the venv wasn't created — go back to 4.2 and check the `pwd` output first; the almost-always cause is that command was run in the wrong folder, so `.venv` got created somewhere else (or not at all).

### 4.4 Activate it

```bash
source .venv/bin/activate
```

**Check your prompt.** It must now begin with `(.venv)`:

```
(.venv) davidkoktavy@Davids-MacBook tradingbot %
```

If you instead saw `Permission denied`, it means step 4.3's check would have failed too — go back and confirm `.venv/bin/activate` actually exists before trying this again. A `.venv` folder that doesn't contain a real virtual environment (for instance, if something interrupted step 4.2) is the only way this specific error happens.

> **You will run this exact command — `source .venv/bin/activate`, from inside `~/Projects/tradingbot` — every single time you open a new Terminal to work on this project.** Forgetting it is the most common reason a later command mysteriously fails. If anything below complains about a missing command or module, check for `(.venv)` at the start of your prompt first, before anything else.

To turn it off later: type `deactivate`.

### 4.5 Install the software

Confirm `(.venv)` is showing, then:

```bash
pip install -e ".[dev]"
```

This downloads a lot and takes a few minutes, with text scrolling past. At the end, look for a line like `Successfully installed ...`. Yellow warning text along the way is normal; red error text is not.

### 4.6 Create your settings file

```bash
cp .env.example .env
cat .env
```

You should see lines including `TRADING_MODE=PAPER`. **Leave that line alone — it's what keeps everything below using pretend money.**

> Never share this `.env` file once it has real values in it — it's where secrets eventually live. For now, with the example values, there's nothing sensitive in it yet.

---

## PART 5: Proving the Setup Works

### 5.1 Run the automated test suite

The project ships with **720 automated tests** that check every part of the system works correctly, with no internet connection and no broker required.

```bash
pytest
```

You'll see a stream of dots — each one a passing test — for maybe 10-20 seconds.

**Look for this exact line at the end:**

```
720 passed
```

If you see that, your entire setup is correct and everything from here on will work. If you see failures instead, something went wrong earlier — check Part 15 (Troubleshooting).

### 5.2 Your first real command

```bash
python -m trading_agent status
```

Expected output looks like:

```
[ MODE: PAPER ]

Health:        DEGRADED
Can trade:     True
  [HEALTHY  ] portfolio: equity=100000
  [HEALTHY  ] kill_switch: inactive
  [DEGRADED ] ai: AI provider unavailable; deterministic strategies only

Kill switch:   inactive
Trading halt:  False
Instruments:   SPY:SMART:USD
Strategies:    ma_crossover
AI provider:   unavailable (deterministic only)
```

Reading it:
- **`MODE: PAPER`** — pretend money. This banner appears on every command on purpose. If it ever says `LIVE`, real money is on the line.
- **`ai: AI provider unavailable`** — expected, since no AI key is configured. The system runs perfectly on its rule-based strategies without one; the AI is optional throughout.
- **`Can trade: True`** — nothing is blocking it.

If this command fails with anything mentioning "No module named trading_agent," you've either forgotten to `cd ~/Projects/tradingbot` first, or forgotten to `source .venv/bin/activate`. Check both.

---

## PART 6: Exploring Safely — nothing here can lose money or place any order

### 6.1 List the strategies

```bash
python -m trading_agent strategies
```

```
Registered strategies (none is claimed to be profitable):

  ma_crossover       v0.1.0    MACrossoverStrategy
  momentum           v0.1.0    MomentumStrategy
  mean_reversion     v0.1.0    MeanReversionStrategy
  trend_following    v0.1.0    TrendFollowingStrategy
```

The software tells you itself: none is claimed to be profitable. Briefly:
- **ma_crossover** — buys when a short-term average price crosses above a longer-term one.
- **momentum** — buys things that have recently been rising.
- **mean_reversion** — bets that an unusually large recent move reverses.
- **trend_following** — buys breakouts above a recent price range.

All four are famous, simple, and widely known — which is exactly why they're unlikely to have a durable edge. If something this simple reliably made money, it would already be arbitraged away.

### 6.2 See the safety limits

```bash
python -m trading_agent risk
```

```
Configured risk limits:

  max_risk_per_trade             0.005
  max_daily_loss                 0.02
  max_portfolio_drawdown         0.1
  max_position_size              0.1
  max_gross_exposure             1.0
  max_open_positions             10
  max_orders_per_minute          20
```

These are fractions, not percentages:

| Setting | Value | Meaning |
|---|---|---|
| `max_risk_per_trade` | 0.005 | Risk at most 0.5% of equity on any one trade |
| `max_daily_loss` | 0.02 | Stop entirely after a 2% loss in one day |
| `max_portfolio_drawdown` | 0.1 | Stop if 10% below the best-ever balance |
| `max_position_size` | 0.1 | Never put more than 10% of equity in one position |
| `max_open_positions` | 10 | Hold at most 10 positions at once |
| `max_orders_per_minute` | 20 | Never send more than 20 orders/minute |

These are **enforced in code**, not suggestions. If a strategy — or an AI, if you ever connect one — asks for something that breaks a limit, it's refused outright, and that refusal is recorded.

### 6.3 Positions

```bash
python -m trading_agent positions
```

`No open positions.` — correct, since nothing has traded.

---

## PART 7: Your First Backtest

A backtest replays historical prices and asks "what would this strategy have done?"

### 7.1 Generate practice data

```bash
python scripts/make_sample_data.py
```

```
Wrote 400 rows of MADE-UP price data to sample_data.csv
This is not real market data. Results from it mean nothing.
```

That warning is honest — it's an artificial mathematical pattern, not real market history, built purely so you can practise the commands.

### 7.2 Run it

```bash
python -m trading_agent backtest --strategy ma_crossover --data sample_data.csv --symbol AAPL
```

You'll see a results summary with numbers like total return, Sharpe ratio, win rate, and — very likely — **warnings underneath them**, something like:

```
WARNINGS: Only 1 trades — below 30; these metrics are not statistically
meaningful; Sharpe of 12.19 is implausibly high for a real strategy;
check for look-ahead bias, survivorship bias, or unrealistic fills
```

**This is the single most important thing to understand from this whole tutorial.** A Sharpe ratio above roughly 3 essentially never happens for a real strategy — genuinely excellent professional funds run around 2. Seeing "12" is not good news; it's the software telling you the result is fake, almost certainly because the practice data is a smooth artificial pattern rather than messy real-world prices. **When a backtest looks amazing, get suspicious, not excited.** Look for the mistake before you look for the payout.

### 7.3 A statistically rigorous second opinion

The backtest above tests one fixed set of parameters. There's a further, sharper check: run the same strategy across a *range* of parameter combinations and ask "given how many combinations I tried, is the best result actually meaningful, or did I just search hard enough to find noise that looks good?"

```bash
python -m trading_agent overfitting-check --strategy ma_crossover --data sample_data.csv --symbol AAPL --grid '{"fast_period":[5,8,10,12,15,20],"slow_period":[20,30,40,50],"atr_period":[10,14]}'
```

This computes something called a **Deflated Sharpe Ratio** — it mathematically corrects for exactly how many parameter combinations you tried. On the artificial practice data, you should see it explicitly refuse to certify the result, usually because it's backed by too few actual trades. That refusal is the tool working correctly, not a bug.

---

## PART 8: Learning From Past Trades

The system can analyse its own trade history for patterns — degrading performance, losing streaks — using plain arithmetic, no AI required:

```bash
python -m trading_agent reflect --strategy ma_crossover --data sample_data.csv --symbol AAPL
```

This always shows deterministic statistics. If you've connected an AI provider (optional, covered in Part 11), it also adds narrative hypotheses on top — but nothing it suggests is ever applied automatically. Any suggested change has to go through a full, separate approval process before it could affect anything, and that process requires a real human's sign-off, not a click.

---

## PART 9: The Monitoring Dashboard

### 9.1 Start it

```bash
python -m trading_agent dashboard
```

```
Dashboard on http://127.0.0.1:8000 (no authentication — bind to localhost)
```

The Terminal will look "stuck" — that's correct, it's running and waiting.

### 9.2 View it

Open Safari or Chrome and go to:

```
http://127.0.0.1:8000
```

You'll see a dark page headed **TRADING AGENT — MODE: PAPER**, in green. **Green means pretend money. If this is ever red with "REAL MONEY," stop and pay attention** — that means real trading is active.

### 9.3 Stop it

Back in the Terminal: **Control+C**.

> **Security note**: this page has no password. `127.0.0.1` means "only this computer" — nothing else on your network can reach it. Never change that to make it reachable from elsewhere.

---

## PART 10: Connecting to Interactive Brokers (Paper Account)

This is optional and can wait as long as you like — everything above works forever without a broker connection.

### 10.1 Get a paper trading account

1. Open an account at **interactivebrokers.com** (a real, regulated application process — identity and financial details required, as with any broker).
2. Once approved, in the Client Portal find **Settings -> Paper Trading Account** and request one. It gets its own separate username and password — **write these down separately from your real login**, since mixing them up is exactly the mistake to avoid.

### 10.2 Install Trader Workstation (TWS)

1. Download **Trader Workstation** for Mac from the IBKR site.
2. Drag it into Applications, open it.
3. **Log in with your PAPER credentials — not your real ones.**

### 10.3 Turn on API access

In TWS: **File -> Global Configuration -> API -> Settings**.

1. Tick **Enable ActiveX and Socket Clients**.
2. Leave **Read-Only API** ticked for now.
3. Confirm **Socket port** shows **7497**.
4. Under **Trusted IP Addresses**, click **Create**, add `127.0.0.1`.
5. Click **OK**, then **fully restart TWS** — quit it and reopen, logging back into paper.

### 10.4 The one number that matters most

| Port | Meaning |
|---|---|
| **7497** | TWS **paper** — pretend money |
| 7496 | TWS **live** — real money |
| **4002** | Gateway **paper** — pretend money |
| 4001 | Gateway **live** — real money |

Check your `.env` agrees:

```bash
cat .env | grep IBKR_PORT
```

Should show `IBKR_PORT=7497`.

### 10.5 Test the connection

With TWS open and logged into paper:

```bash
python scripts/smoke_test_ibkr.py --symbol AAPL
```

This checks connectivity without placing any trade. Look for `Passed: 10   Failed: 0` (or similar) at the end. If anything fails, do not proceed further until it's resolved — this test exists specifically to catch broker-connection problems before they matter.

---

## PART 11: Connecting an AI Provider (Optional)

Everything above runs perfectly with no AI at all — the system defaults to its rule-based strategies. If you want the AI advisory layer (which can propose trades but never bypasses any risk limit, and never sizes its own positions), add your API key to `.env`:

```
AI_PROVIDER=anthropic
AI_MODEL=<Sonnet 5>
ANTHROPIC_API_KEY=<your key>
```

Nothing else changes. Re-run `python -m trading_agent status` afterward and the `ai:` line should now say `available` instead of `unavailable`.

---

## PART 12: Feeding It Real-World Context (Optional)

If you're tracking something in the news — a climate event, a policy shift, anything — you can record it as labelled context for the AI to consider. **This is not a live news feed; you type in what you're tracking, and it's always clearly marked as an unverified hypothesis, never a fact or an instruction:**

```bash
python -m trading_agent macro add --name "Example event" --category CLIMATE --stance MIXED_UNCERTAIN --description "A hedged, uncertain note about what you're tracking." --sectors AGRICULTURE --confidence 0.4 --source "Where you read this" --expires-in-days 90
```

```bash
python -m trading_agent macro list
```

Every entry requires an expiry date on purpose — stale narratives shouldn't linger as permanent context.

---

## PART 13: Reading the Audit Trail

Every decision the system makes — including deciding *not* to trade — gets recorded with full reasoning.

```bash
mkdir -p logs
python -m trading_agent simulate --symbols AAPL --max-cycles 5
python -m trading_agent explain --audit-log logs/decisions.jsonl
```

You'll see a full narrative per decision: market conditions, which strategy signalled what, every safety check that ran and whether it passed, and the eventual outcome. This is what lets you answer "why did it do that?" months later with evidence instead of guesswork — including, importantly, "why did it *not* trade," which is usually the more informative question.

---

## PART 14: The Emergency Stop

```bash
python -m trading_agent kill-switch --reason "Something looks wrong"
```

This blocks all new orders immediately.

**There is deliberately no "undo" command.** If the system stopped itself, a human should look at why before deliberately restarting — not press a button and hope. The fastest real stops, in order:

1. **Control+C** in the Terminal running the agent.
2. **Quit TWS.**
3. Log into IBKR directly and cancel/close manually.

Know these before you need them.

---

## PART 15: Troubleshooting

### "No such file or directory" on `cd` or `ls`

You're not where you think. Run `pwd`, compare it to what's expected at that step, and use `cd ~/Projects/tradingbot` to get back to the project root.

### "does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found"

You're one folder too deep — almost always inside the small inner `trading_agent` folder described in Part 3.5. Run `cd ..`, then `ls pyproject.toml` to confirm you're back at the real root.

### `.venv/bin/activate: Permission denied` or "No such file or directory"

The virtual environment was never actually created, usually because `python3.12 -m venv .venv` was run from the wrong folder. Go back to Part 4.2, confirm `pwd` first, then re-run it, then re-check with `ls .venv/bin/activate` (Part 4.3) before trying to activate again.

### Doubled paths like `/Users/name/Users/name/...`

You combined `~` with a full `/Users/...` path in the same command. Re-read Part 1.5. Use one or the other, never both.

### `command not found: python3.12`

Either the python.org installer didn't finish, or you didn't open a fresh Terminal window afterward. Close the Terminal completely, reopen it, try again. If it still fails, use the long path from Part 2.3.

**If you switched from bash to zsh (Part 1.3) *after* installing Python**, there's a specific extra cause worth checking: the python.org installer configures whichever shell was active at the moment you installed it. If Python was installed while you were in bash, zsh may not know where to find it yet. Two fixes, try in order:

1. Open **Finder → Applications → Python 3.12**, and look for a file named **`Update Shell Profile.command`**. Double-click it, let it run, then close and reopen Terminal.
2. If that file isn't there, add the path manually:
   ```bash
   echo 'export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"' >> ~/.zprofile
   ```
   Then close and reopen Terminal, and try `python3.12 --version` again.

### "No module named trading_agent"

Two possibilities, check both: (1) you're not in `~/Projects/tradingbot` — run `pwd` to check; (2) the virtual environment isn't active — check for `(.venv)` at the start of your prompt, and if it's missing, run `source .venv/bin/activate`.

### Tests fail

```bash
pip install -e ".[dev]" --force-reinstall
pytest
```

### Smoke test connection fails

Check, in order: TWS is actually open and logged in; API access is enabled in its settings; the port in `.env` matches TWS's setting (both 7497); `127.0.0.1` is in Trusted IP Addresses; you fully restarted TWS after changing settings.

### Starting completely fresh

```bash
cd ~/Projects
rm -rf tradingbot
```

Then repeat from Part 3.

---

## PART 16: Command Reference

Always with `(.venv)` showing, always from `~/Projects/tradingbot`.

| Command | What it does |
|---|---|
| `python -m trading_agent status` | Health check and current state |
| `python -m trading_agent strategies` | List available strategies |
| `python -m trading_agent risk` | Show safety limits |
| `python -m trading_agent positions` | Show current holdings |
| `python -m trading_agent backtest --strategy X --data Y.csv` | Test a strategy on historical data |
| `python -m trading_agent overfitting-check --strategy X --data Y.csv --grid '{...}'` | Statistically correct a parameter search for overfitting |
| `python -m trading_agent reflect --strategy X --data Y.csv` | Analyse past trades for patterns |
| `python -m trading_agent macro add / list / remove` | Manage global-event context (not live news) |
| `python -m trading_agent simulate --max-cycles N` | Run with a pretend internal broker |
| `python -m trading_agent reconcile` | Compare local records against the broker |
| `python -m trading_agent explain --audit-log logs/decisions.jsonl` | Read the decision history |
| `python -m trading_agent dashboard` | Start the web dashboard |
| `python -m trading_agent kill-switch --reason "..."` | EMERGENCY STOP |
| `python -m trading_agent migrate` | Apply database updates (advanced) |
| `pytest` | Run all 720 self-tests |
| `python scripts/make_sample_data.py` | Create practice price data |
| `python scripts/smoke_test_ibkr.py --symbol AAPL` | Test the broker connection |

Terminal basics:

| Command | What it does |
|---|---|
| `pwd` | Where am I? — use this constantly |
| `ls` | What's here? |
| `cd foldername` | Go into a folder |
| `cd ..` | Go up one level |
| `cd ~/Projects/tradingbot` | Go straight to the project |
| `source .venv/bin/activate` | Turn on the environment (every session) |
| **Control+C** | Stop a running program |

---

## PART 17: A Sensible Pace

**Week 1-2:** Backtests only. Try every strategy on different practice data. Get comfortable watching results swing wildly with small changes — that instability is itself the lesson.

**Month 1:** Connect to the IBKR paper account, get the smoke test fully passing. Watch the dashboard. Read audit trails. Change nothing yet.

**Months 2-4:** Let it run in paper mode. Read the audit trail regularly — which risk limits keep blocking trades? Does behaviour match what you expected?

**Months 4+:** If still interested, read the actual code, starting with `risk/risk_engine.py`, until you can explain every safety check before trusting any of them.

**Most people should stop at paper trading, indefinitely.** That's not failure — learning a strategy doesn't work without paying tuition for the lesson is the most valuable thing this software can give you.

---

## PART 18: Never / Always

**Never:**
- Set `TRADING_MODE=LIVE` before months of paper trading
- Raise a risk limit because trades keep getting blocked — that's the system working correctly
- Trust a backtest with under 100 trades, or a Sharpe ratio above 3
- Trade money you need for anything else
- Expose the dashboard beyond `127.0.0.1`
- Share your `.env` file once it has real values

**Always:**
- Check the mode banner before every session
- Run `pwd` when in doubt about where you are
- Read the audit trail after anything surprising
- Run `pytest` after any change
- Know where the emergency stop is
- Believe the warnings the software gives you

---

## Final Word

You now have a complete, working trading system with more built-in safety discipline than most amateur setups ever get. It can't give you an edge in the markets — nothing can promise that — but it can make sure that whatever you do, you do it with proper position sizing, hard limits that actually stop it, a full record of why, and an emergency stop you know how to reach.

Go slowly. Verify every step. Be suspicious of good-looking results before you're excited by them.
