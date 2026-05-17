# TWN2N v8 anchor sweep — case (b) on GitHub Actions

This repository is a **remote compute target** for case (b) of the Monte Carlo
sweep of the TWN2N anchor architecture, distributed across the workstation,
the cluster, and the free GitHub Actions matrix runner. It does not contain
the rest of the paper — only the script that runs the sweep, a workflow that
slices it across 42 parallel jobs, and a small merge step that concatenates
the per-job CSVs into a single `case1_v8_anchor_caseb_merged.csv` artifact.

## What runs here

The sweep covers, for case (b) (single-shaker broadband chirp):

* SNR ∈ {5, 10, 15, 20, 25, 30, 35} dB
* N ∈ {1, 2, 4, 6, 8, 16}
* `n_z` ∈ 1..2N for N ≤ 6, 12 evenly spaced values for N ∈ {8, 16}
* `p` ∈ {1, 2, 3, 4, 5, 6}
* 3 trajectories per cell concatenated for training (anchor architecture: one
  scalar output per cell)
* 1 random system per (case, N) seed

Total: **2 100 cells**. Sliced into 42 GitHub Actions jobs (one per `(SNR, N)`
pair). Each job runs at most 72 cells, so even on the 2-vCPU runners the
slowest slice stays inside the 6 h job cap with comfortable margin.

The full architecture / sampling / metrics are documented in the paper draft
this script lives next to in the main project.

## Run it

The workflow is triggered manually:

1. Open the *Actions* tab of this repo.
2. Pick **case-b sweep**.
3. Click **Run workflow**.

The matrix expands to 42 jobs; GitHub schedules up to 20 of them concurrently
on a public repo, so the whole sweep finishes in roughly 4 h of wall time
once it's queued.

Each job uploads its slice of `case1_v8_anchor_results.csv` as an artifact
named `csv-snr<SNR>-N<N>`. After every slice finishes a final `merge` job
downloads all of them and writes `case1_v8_anchor_caseb_merged.csv`, the
artifact you'd actually pull down at the end.

## Pull the merged CSV

```bash
gh run download <run-id> -n merged -D ./merged
# or via the Actions UI: Artifacts -> merged -> Download
```

## Re-running a single slice

If only one of the 42 jobs failed (e.g. transient runner issue), you can
re-trigger just that slice locally with the same flags the workflow uses:

```bash
python case1_twn2n_v8_anchor.py --cases b --snr-only 15 --n-modes-only 8 --restart
```

The slice writes to `case1_v8_anchor_results/case1_v8_anchor_results.csv`,
which you can then commit + merge with the other 41 slices by hand.
