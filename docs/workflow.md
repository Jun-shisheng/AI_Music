# Batch workflow

How a batch actually gets run, and why each step exists. The rules below are the
ones that survived contact with real deliveries; each carries the failure it was
written to prevent.

## 0. Before the GPU is touched

- **Free the VRAM.** Anything resident (an image-generation UI, another model
  server) must be closed. A resident process holding 10 GB will starve a render
  into failure or into a 4× slowdown.
- **Write the brief as files, not as command-line arguments.** One text file for
  the lyrics, one for the style caption, one matching title per song. Long
  captions passed through a shell → PowerShell → process boundary lose
  characters; files do not.
- **Keep every title in its own process.** A long-lived process accumulates a
  fragmented CUDA allocator and eventually dies on the next large single
  allocation, mid-batch. One process per title costs one model load and removes
  the whole class of failure.

## 1. Generate

Each title is rendered **more than once, with a different seed**, because the
model samples things you cannot control — how long the piece is, whether the plan
opens with an instrumental intro, and how much it repeats itself. Candidates are
cheap relative to the cost of accepting a bad take.

Running unattended means the loop must be:

- **resumable** — a title whose take already passed is skipped, so an interrupted
  run continues instead of redoing the night;
- **time-budgeted** — the budget is checked between titles, so the run ends when
  it should rather than when the machine dies;
- **disk-visible** — state goes to a `RUN_STATUS.txt` in the output folder, so an
  interrupted session is recoverable by reading a file instead of a chat log.

## 2. Measure

Metrics are computed per take, never per batch:

| Metric | Meaning | What it catches |
|---|---|---|
| duration | model-chosen, not requested | a plan that samples a 4:29 piece in a 3:30 window |
| opening level, lift | dB | a take that is loud from bar one |
| climax position | 0-1 on the timeline | a peak at 0.3 — the song has nowhere to go |
| body level under its own peak | dB | a mix that sits at full volume throughout, heard as "flat" |
| first sung word | seconds | a 30-second instrumental intro before the lyric |
| diction ratio / lines recovered | against the intended lyric | the model singing something else, or dropping a verse |
| clicks / dropouts | sample-level, diagnostic | broken files (not a defect verdict — see below) |

Two rules about these numbers:

- **A gate is only for defects that are unambiguous** — wrong length, no audio at
  all, clipping, near-silence. Everything aesthetic ranks candidates instead of
  rejecting them, because a proxy that cannot hear should not be able to veto a
  take.
- **A click count is not a defect metric in percussion-heavy music.** Drums are
  transients. One measured delivery spanned a 100× click-rate range between
  titles, and none of it was a defect.

## 3. Deliver

Delivery is mechanical so nothing is forgotten at the end of a long run:

```
<YYYY-MM-DD>_<name>/
    <slug>_<engine>.wav     # normalised, source-clean reference kept beside it in the batch folder
    <slug>_<engine>.txt     # section tags once; each translated line under its source line
    manifest.json           # every take's own numbers, so the folder explains itself
    <YYYY-MM-DD>_<name>_review.md   # the listening sheet
```

- **One folder per delivery.** Songs and their lyrics travel together, paired by
  identical file stems, so a lyrics file can never be orphaned from its audio.
- **The lyrics header carries the take's own numbers** (engine, structure, lift,
  climax position, first-sung time, diction ratio). Whoever opens the text file
  knows what they are listening to.
- **A listening sheet is not a metrics dump.** It says how to read each number and
  what to confirm by ear, in the listener's own language.
- **Sanity-check the sheet against the files on disk.** Twice, a sheet named files
  that had been discarded by a gate; both times the fix was to require the
  promoted WAV to exist before a row is printed.

## 4. Listen, then iterate

The loop closes with a human, and the loop is only as good as how their feedback
is recorded. "It sounds flat" is not actionable; **"1:12-1:18 in track 03"** is.
Turn the feedback into timestamps, measure exactly those seconds against the rest
of the track, and only then decide whether the fix belongs in the prompt, in the
generation settings, or in the model itself.

## Operating notes

- **Time the stages, not the run.** Measured on one 12 GB card: loading ~50 s,
  token generation at ~10 it/s, codec decode the dominant cost, normalisation and
  MP3 encoding effectively free. Optimising the wrong stage is easy.
- **Budget for variance.** The same brief and settings produced takes between 11
  and 31 minutes. A plan that samples a long piece costs both wall-clock and
  quality.
- **Watch free RAM, not just VRAM.** A resident model can take a machine from
  17 GB free to under 2, and renders slow down by 4× without any audio change.
- **`.bat` files: ASCII only.** `cmd.exe` reads them in the OEM code page, and
  UTF-8 bytes inside a `rem` comment can decode onto a command separator. Same
  trap for a BOM-less UTF-8 `.ps1` under PowerShell 5.1.
- **Never `git add -A` in a working tree that also holds audio, model weights and
  run logs.** Name the files you mean.
