# RIR review and repair — 2026-09-19 / 0.2.13

## Compared versions

- GitHub main: `fee440327bf47949759f1ce331f579f82ad417ca`, declared version 0.2.11.
- Local export: `E:/Code_Github/python_acoustic_capture_refactored_0.2.12/python_acoustic_capture`.
- Repair branch: `codex/rir-repair-20260919`. Original export and main branch remain unchanged.

The export splits RIR deconvolution, timing and quality logic into dedicated modules. It also fixes GUI initialization/config snapshots, validates short callback recordings, preserves real captured frame counts on cancellation, checks common RIR crop coverage, and saves processed RIR as floating-point WAV. Those changes were brought into the current checkout. Its tests/configs were reconciled with the fixed-count and seeded-pairing implementation; stale tests referencing removed Cartesian-pairing APIs were replaced by the export's current tests.

## Additional repairs

1. Windows audio operations initialize COM in their owning thread and balance cleanup even after exceptions. Stream start/stop/close remain on the owning thread. This addresses an initialization precondition, not a guarantee that every vendor ASIO driver works.
2. Direct-arrival detection selects a peak inside the first detected energy packet, rather than searching a 2.5 ms interval that can select a stronger later reflection. This remains a detectable-packet estimate, not proof of geometric direct-path arrival.
3. GCC-PHAT versus low-frequency group-delay disagreement, unreliable correlation, and disagreement with first-packet timing require review. Reflections and microphone phase response can legitimately make these estimates differ; the tool must not silently certify them as true acoustic TDOA.
4. Full-response convolution reconstruction is now the default acceptance policy, using absolute MSE/NMSE and correlation without fitting away amplitude errors. Repeatable nonlinear distortion is rejected. A deliberately truncated early RIR can still use explicit `reconstruction_policy: diagnostic`, which does not claim full-response reconstruction acceptance.
5. Repeated takes are aligned using one common shift for every microphone. There is no separate per-microphone peak alignment or peak normalization. Only accepted takes within one experiment contribute to its arithmetic mean. Two inconsistent takes are not automatically accepted.
6. GUI completion now displays nested RIR quality issues and records them in the checklist error field. Acquisition completion alone is not a training-quality certificate.
7. Ten-pair verification uses the actual seeded pairing plan and no longer deletes an existing results folder. Portable default configuration and the 2000-pair stress tool use current pairing fields.

## Verification evidence

- Automated suite: **137 passed**, including Tk GUI construction/workflow tests, callback failures/short streams/cancellation, multi-channel preservation, repetition consensus, crop coverage, reconstruction rejection, and actual Windows worker-thread COM lifecycle.
- MATLAB **R2024b**, actual local execution of `sweeptone` and `impzest`: four cases (0.5 s / -12 dBFS, 8 s / -18, -30 and -60 dBFS), 48 kHz, 40–22000 Hz. Maximum ESS difference **1.4886e-8**; maximum IR difference **2.7669e-8**. Both microphones include known delayed direct impulses and later echoes. Tests cover these parameter sets, not every possible sweep setting or MATLAB release.
- Strong-reflection counterexample: true direct peaks at samples 173 and 180, stronger reflections at 221 and 260. Repaired extraction returns 173/180 and **7 samples** relative delay; ambiguous correlation timing is flagged.
- Simulated complete five-take RIR capture through the CLI: five accepted takes, mean RIR quality `pass`, relative first-packet delay **7 samples**, worst median reconstruction NMSE about **-20.74 dB**, minimum correlation about **0.996**.
- Simulated ten speech pairs: **10 label rows / 10 supervised pairs** successfully compiled, with target-only, interferer-only and mixture captures. This is regression coverage, not physical speaker/microphone validation.

Local generated evidence (ignored by Git):

- `outputs/rir_matlab_float_20260919/matlab_comparison.json`
- `runs/20260919_173134_simulated_demo_rir/manifest.json`
- `outputs/verification_10_pairs_20260919/`

Reproduce MATLAB comparison from the project directory with this project's `src` on `PYTHONPATH`:

```text
python tools/verify_rir_matlab.py outputs/new_matlab_check
matlab -batch "addpath('tools'); verify_rir_matlab('outputs/new_matlab_check')"
python -m pytest -q
python -m acoustic_capture rir configs/simulated.yaml
python tools/verify_ten_supervised_pairs.py --output-dir outputs/new_speech_check
```

## Still requires laboratory hardware acceptance

No RME/ASIO device was enumerated on this machine during the review. Therefore actual RME duplex operation, physical microphone TDOA, driver cancellation under load, and long hardware recordings have **not** been verified. No physical playback was performed for the above tests.

Use a shared-clock ASIO duplex device in the lab; confirm channel routing and absence of input/output overruns, then compare repeated measurements against a wired loopback or independently known reference. Do not equate the largest RIR peak, low-frequency group delay, or GCC maximum with guaranteed geometrical direct sound. Independent playback/recording devices do not share a clock merely because both are wired.

The existing portable EXE/ZIP has **not** been rebuilt. It does not include these source fixes. This report does not certify the old binary. Changes are local until explicitly pushed.
