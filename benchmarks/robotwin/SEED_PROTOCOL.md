# RoboTwin fixed inputs v2

SDK 0.7.0b51 uses evaluation protocol
`robotwin2-platform-native-step-lim-v2`. The Policy observation/action contract
remains `robotwin2-v1`; this revision changes the fixed evaluation inputs.

Both suites contain the manifest's ordered 50 tasks, each with exactly 100
expert-valid episodes. The archive is committed here. `build-runtime.sh` copies
it into the install directory, verifies its SHA256 against
`benchmark-manifest.json`, extracts it into `seed-cache/`, and builds one
generation per suite under `seed-generations/`, failing unless each generation's
digest equals the manifest's. Formal runs validate the generation again before
evaluating. No other cache selects the inputs.

| Input | SHA256 |
| --- | --- |
| `seed-cache-v2.tar.gz` | `61a94edca4b2d9a7a75a9888ecc5df903e6ff956b2dada303c8eb6c38f1e5b80` |
| v2 `demo_clean` generation | `1b4aa12bcbad2fa3e926f5109a89631ba5043e89e1cbdab488b78ef85629caca` |
| v2 `demo_randomized` generation | `b4b0c4c2bfe12d49c5ed9c7f8b5ec07a3cc66e1d020f08515cd5e3ba779b09c2` |

The clean entries were recovered from the preserved 2026-09-07 native-input
snapshot. They reproduce v1 digest
`247acb6197a6cd49f67511b3ba11a5e296f67ab95f6571f25cfd6d250724a3de`.
Only the protocol identity changes their v2 generation digest.

The old randomized pool could not be recovered. The 2026-09-02 inventory records the original
`b447fe908298233734c580ccbe2cd84fda1d9aa90aa0cb08245561946b39145c`
generation's paths and sizes, but not its contents. That pool became
`d67685546747d3df6ecc13719fb24fd1fc51104e52d006a1f0ded7e34c7b0ab9`
after the pinned runtime annotation patch. The 2026-09-29 RoboTwin backup
explicitly contains only smoke seeds, not a complete formal pool.

The replacement randomized entries were generated on 2026-10-13/14 with the
committed native expert scanner, ascending candidates from zero for each task,
and the first 100 expert-valid entries. No Policy success results were used to
select seeds. Its content under the v1 identity computes to
`a8a00f2b1fe1a90751212397727c80d7f3cd754b44e7c06460187c45e5a26b33`,
which differs from the missing v1 pool. Randomized v1 and v2 scores therefore
must be reported separately.

Source identities remain pinned in `benchmark-manifest.json`: RoboTwin base
`c3ddfa8b97d5519efa828b075999bd0006778e5e`, prepared runtime
`014728c265a7f06c8b7c2ace7e94188c3253e43a`, and scanner identity
`c8d681bebecf67f8f0c2e1dd0a67392f9fea4431`. Scanner SHA256 is
`6a6e94fdf001ceec5d53a30fac9f075d754b4fdbf30dc48a41709455a74cf20e`.
The recorded native scanner environment is Python 3.8.20, PyTorch 2.4.1+cu121,
CuRobo v0.7.8, and Ada compute capability 8.9. The frozen archive, rather than
rerunning this environment, is the source of future evaluation inputs.

From SDK 0.7.0b52, one autotune candidate can use the full 1,800-second total
autotune budget. The previous 900-second candidate limit interrupted the
randomized baseline after nine of ten representative episodes on a cold RTX
4090 run. Task failures can consume the full native step limit, so a complete
baseline needs this longer allowance. The total budget, native step limits,
fixed inputs, and success criteria remain unchanged.

This input recovery does not change recorded scores.
