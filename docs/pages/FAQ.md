## Why use this instead of a chatbot?

A big chatbot knows far more and writes better. Ask Physics does a narrower job in a way
you can check:

- **Every number is computed, not written.** Symbolic algebra with units does the math,
  so there are no slipped digits or invented results.
- **You see the work.** The equation and its source, every value and where it came from,
  every assumption.
- **It says when it isn't sure.** A low confidence score or a PARTIAL answer instead of
  a confident guess.
- **It's free and it's yours.** It runs on your computer, needs no account, sends your
  questions nowhere, and is MIT licensed.

Think of it as a symbolic physics calculator that reads plain English.

## Does it send my questions anywhere?

No. The CLI runs entirely on your machine, and the website runs the same code inside your
browser.

## Why did my question come back PARTIAL?

A stage couldn't finish, and the card says which one. Most often the equation isn't in the
database yet ("retrieve"), or no plan could be made to work ("plan" or "sanity_check").
[Asking good questions](Asking-Good-Questions) has phrasings that help.

## Why is the confidence low when the answer looks right?

The checks found something worth a second look: an unused value, a question that fits two
look-alike equations, or a result outside the usual range. The caveats say which.
[Reading an answer](Reading-an-Answer#confidence) explains the formula.

## Does the website use the Fermi models?

Not yet. The website runs the real package in your browser, with the same retrieval,
math, and checks, but PyTorch can't run there, so it uses the built-in fake model, which
only plans "dropped from a height" questions. Anything else comes back PARTIAL, saying
which stage it couldn't do. Running the models in the browser is planned once trained
weights ship. For everything else, use the CLI.

## Can it do Fermi estimates?

It classifies them and has a table of reviewed estimates ("a rubber duck masses about
25 g"). Estimates with proper low and high ranges arrive in v0.7.

## Why are the models so small?

So they run fast on a laptop, and because their job is narrow: read the question, pick
from a short list, copy numbers, write a few sentences. Constrained decoding keeps them
honest, and the math is never theirs to get wrong.

## Where are the trained models?

The installers and Homebrew download them when they install, and the first `askphysics ask`
downloads them if they are still missing (a pip install, say, or a download that failed). They
live in `~/.cache/askphysics/models/` (or `$ASKPHYSICS_MODEL_DIR`; Homebrew keeps them in its
`var` folder). To build them yourself, see [Training the models](Training-the-Models).

## Why did it ask me to download celeste?

celeste is the biggest model (about 240 MB) and only tries a question solem couldn't plan,
so it never downloads without your say-so. Answer `y` for this once, `a` to always allow it,
or `never` to stop being asked; delete `preferences.json` in the models folder to reset that.
Without a terminal (`--json`, scripts, the website) it is skipped instead of asked about.

## Something's wrong. Where do I report it?

Open an issue at
[github.com/shankar-sachin/ask-physics/issues](https://github.com/shankar-sachin/ask-physics/issues)
with the question you asked and the output of `askphysics ask --json "..."`. Security
problems go through
[`SECURITY.md`](https://github.com/shankar-sachin/ask-physics/blob/main/SECURITY.md).
