# AutoDL shared-runtime inode compaction

AutoDL file storage provides 200,000 inodes. The expanded RoboCasa, RoboTwin,
and LIBERO+ runtime roots consume most of that quota. Their immutable runtime
trees belong in one zstd level-3 tar archive per benchmark; each evaluation
extracts its selected runtime into the instance-local 100 GB system-disk
expansion. A checksum-pinned zstd 1.5.7 binary and its two runtime libraries
remain in the shared thin-tool directory, adding only a handful of inodes.

RoboTwin archives only the two production inputs named by the runtime
contract: `root/miniconda3` and `root/RoboTwin`. Retired `root/starVLA`, old
results, engine wrappers, and test artifacts account for about 9.9 GB and are
excluded from every instance cold start.

## Measured result

In the first production rollout, compaction reduced shared-storage inode usage
from about 94% to about 1% of the 200,000 quota (a net reduction of roughly
185,000 inodes), with complete recovery archives retained.

## Historical inode forecast

The counts in this section are historical forecasts from the original
whole-batch design. Incremental rollback journals and retained recovery
archives add their own entries. Production savings are recorded after each
batch's finalization, when its retained expanded tree is released.

An inventory measured approximately 84,658 inodes in RoboCasa runtime, 54,489 in
RoboTwin runtime, 37,459 in LIBERO+ runtime, and 8,584 in the obsolete shared
Control Runtime: about 185,190 inodes in total. The compact representation
needs about 24 inodes for three archives and manifests, three runtime-root
symbolic links, the helper/wrapper layer, and the checksum-pinned zstd tool.

The isolated full-data shadow E2E measured 175,446 inodes before compaction and
379 after compaction: **175,067 released inodes (99.784%)**. Applied to the
original production inventory, the exact source prefixes removed by apply hold
184,905 inodes. The final layout adds 24 inodes for root links, archives,
manifests, wrappers, and zstd tools. Expected production usage is therefore
roughly **1,778 / 200,000 (0.9%)**, leaving about **198,222 free inodes**. The
earlier shadow delta remains the directly measured reduction for the isolated
three-benchmark data set.

The newer migration inventory provides a second, conservative regular-file
view: 163,009 regular files disappear from the three expanded roots and shared
Control Runtime. Including the compact files, the complete benchmark tree is
projected to move from 164,410 regular files to about 1,415, a 99.1% reduction.
The maintenance gate records authoritative `df -i` and full tree counts before
and after apply.

The shared path `runtime/root` becomes a symbolic link to the corresponding
instance-local path. Existing Conda prefixes therefore continue to resolve
through their original absolute paths while every regular runtime file lives
on the instance system disk.

## Controlled rollout

Use a clean checkout of the exact RoboScope Kit commit intended for the
shared launchers. Keep the AutoDL shared disk mounted at
`/root/autodl-fs`.

First collect the read-only preview:

```bash
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --preview liberoplus
df -h /root/autodl-fs
df -i /root/autodl-fs
```

The rollout is incremental. Select exactly one benchmark for every operation, in the order LIBERO+,
RoboCasa, RoboTwin. The commands below use LIBERO+ as the first batch.

Create and byte-verify its runtime and recovery archives while the expanded
tree remains available:

```bash
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --prepare liberoplus
```

The apply phase requires a benchmark maintenance window. Stop the
benchmark worker first, then identify every instance that can still access the shared runtime. Wait until
that set is empty; submissions received during the window remain queued until
the worker resumes. It verifies every archive checksum, entry count,
regular-file byte count, and tar-to-source comparison before changing any
runtime. It preserves the selected live
`prepare-runtime.sh` as `prepare-runtime-inner.sh`, installs a thin archive
wrapper and replaces the selected runtime root with its local-runtime symbolic
link. The original expanded root is retained under that benchmark's
`.roboscope-compaction/expanded/root` until acceptance. The wrapper keeps the live manifest, SDK
version, launcher, and benchmark preparation semantics byte-for-byte intact.
This avoids requiring existing images to change their
manifest reference for the inode compaction rollout.

```bash
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --apply liberoplus
df -i /root/autodl-fs
```

Resume dispatch after the selected path's structural checks, then run its real
end-to-end evaluation. Exercise rollback with dispatch paused and affected
tasks drained, verify the original path, reapply, and repeat compact-path
acceptance before reclaiming the retained tree:

```bash
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --rollback liberoplus
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --apply liberoplus
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --finalize liberoplus \
  /root/autodl-fs/benchmarks /root/rollout-evidence/liberoplus.acceptance
```

The acceptance file records `benchmark`, `archive_sha256`, `cutover_id`,
`end_to_end_status=passed`, `rollback_reapply_status=passed`, `task_id`,
`evidence_file` (absolute local path), and `evidence_sha256`. It refers to the
latest cutover and the operator-verified end-to-end evidence. Finalization
verifies that record and full recovery material before reclaiming the tree.
The full recovery archive and original preparation bytes remain available
for `--rollback` after finalization as well.

After each batch is accepted and reclaimed, advance to the next benchmark.
Once all three are finalized and image-local control execution is proven,
reclaim the shared Control Runtime separately:

```bash
benchmarks/starvla_platform_runtime/compact-shared-runtimes.sh --finalize shared-control
```

Its recovery archive remains in `.roboscope-control-recovery`; restore it with
`--rollback shared-control` during a controlled maintenance window.

Each evaluation performs one streaming extraction. The zstd frame checksum is
verified by the decoder, the extracted root is atomically installed, and the
preserved benchmark-specific inner preparation verifies its required runtime
paths. Full SHA-256, tar-to-source, entry-count, and byte-count verification
stays in the one-time prepare/apply maintenance gates, so instance cold starts
avoid a second integrity scan and two complete local-tree walks.

## Measured startup cost

On 2026-09-01, an AutoDL instance measured archive extraction from
AutoDL file storage into its 100 GB expanded system disk. Every
source archive stayed read-only. Timing starts before `tar` and ends after
`sync`, so the result includes local-disk persistence. The container kernel
kept its normal cache policy.

These four rows describe pre-existing archives and intentionally retain their
historical, mixed formats; they establish real workload sizes rather than a
codec comparison.

| Benchmark | Historical format | Shared archive | Archive size | Extracted entries | Elapsed | Archive throughput |
|---|---|---|---:|---:|---:|---:|
| LIBERO | gzip | `libero-runtime.tar.gz` | 884 MiB | 44,516 | 32.96 s | 26.83 MiB/s |
| CALVIN | raw tar | `calvin-env.tar` | 1.86 GiB | 35,675 | 22.54 s | 84.34 MiB/s |
| RoboCasa | raw tar | `starVLA-env.tar` | 6.69 GiB | 69,667 | 71.52 s | 95.77 MiB/s |
| LIBERO+ | raw tar | `LIBERO-plus.tar` | 8.99 GiB | 478,816 | 94.09 s | 97.87 MiB/s |

The raw archives converge around 84–98 MiB/s. LIBERO trades shared
I/O for gzip decompression and expands 2.52 GiB in 32.96 seconds. In contrast,
copying CALVIN's 145 MiB / 934-entry source tree as individual filesystem
entries reached only 8.39 MiB/s and took 17.28 seconds. Sequential archives
therefore address both inode consumption and cold-start metadata overhead.

### Codec selection

A second measurement used the same RoboCasa 6.69 GiB / 69,667-entry
payload for every codec. The input stream was capped at 100 MiB/s, below the
measured 119 MiB/s shared-disk sequential-read rate, and timing ended after
`sync`.

| Codec | Archive size | Build time | Extract + sync | Change vs raw |
|---|---:|---:|---:|---:|
| raw tar | 6.69 GiB | — | 92.38 s | baseline |
| gzip level 1 | 3.56 GiB | 173.39 s | 88.21 s | -4.5% |
| zstd level 1 | 3.27 GiB | 27.10 s | 73.51 s | -20.4% |
| **zstd level 3** | **2.80 GiB** | **24.06 s** | **62.65 s** | **-32.2%** |
| zstd level 10 | 2.44 GiB | 28.88 s | 68.89 s | -25.4% |

Level 3 is the single archive format for all newly compacted runtime roots. It
beats gzip and raw tar on both one-time archive creation and per-instance cold
start. Level 10 transfers fewer bytes but spends enough extra decompression CPU
to finish later.

The full RoboTwin root contains 32.23 GiB. Selecting only its production
subtrees produced a 23.03 GiB raw payload and a real 16.97 GiB zstd archive.
The native wrapper completed in 127.194 seconds; wrapper plus the successful
local-runtime GPU quick test completed in about 176.846 seconds (2:56.85).

## Runtime verification

Create one fresh AutoDL instance for each affected benchmark and run its real
quick-test path. Record these artifacts:

- shared-disk inode usage before and after apply;
- archive SHA-256, entry counts, and byte counts;
- RoboCasa, RoboTwin, and LIBERO+ quick-test results;
- instance-local runtime paths resolved through each shared symbolic link;
- power-off, shutdown, and release completion for every direct API instance.
