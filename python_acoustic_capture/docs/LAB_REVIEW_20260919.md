# Laboratory workflow review — 2026-09-19

## Scope and evidence

Reviewed S1 ESS generation, deconvolution, common microphone crop and repeat
averaging; thread-affine continuous audio sessions; S23 playback routing,
paired sample layout and labels; S4 stage prompts and recording quality.
The full regression suite passes (163 tests), including simulated workflows,
Tk event handling, interruption/resumption, RIR timing and acceptance tests.
This is not an RME hardware certification: this computer currently enumerates
MME, DirectSound, WASAPI and WDM-KS, but no ASIO device. No physical capture was
performed in this review. Desktop screenshot inspection was obstructed by
other windows and was not accepted as visual QA. Portable EXE was not rebuilt.

## Changes

- RME template now includes all three S23 stages, including interferer-only.
- Three-stage S23 recordings enable acoustic additivity QC by default unless
  an explicit metadata quality_gate overrides it. Default requirements:
  residual <= -6 dB and correlation >= 0.70, across all microphones.
  Failing recordings and labels remain available but are not training-ready.
- Non-finite QC values cannot pass an enabled supervision gate.
- S4 now checks clipping, silent channels, exact duplicate channels and
  non-finite samples; labels retain reasons and mark quality_passed=False.
  This is signal-integrity QC, not speech recognition or a reading assessment.
- S4 uses one corpus line per round. Target-only and mixed display the same
  line; interferer-only requests silence. Advance only on the next round,
  wrapping at the end of the corpus. The text is 24-point bold with reflow.
- Corrected mono quality metrics interpreting samples as separate channels.
- UI no longer promises that every captured pair is directly training-ready.

## Confirmed design and remaining limits

- S1 keeps a common time grid across microphones and averages accepted takes
  within one run only. Independent wearing/angle/height tests are not averaged.
  Direct-arrival ambiguity and reconstruction failures remain QC diagnostics;
  equal-looking peaks alone do not prove the microphones have zero delay.
- ASIO stream construction, start, stop and close remain on the owning worker
  thread, with COM initialized there. GUI Stop only requests cancellation.
  A native driver call that itself hangs cannot be interrupted by Python's
  callback watchdog; real driver/USB testing remains necessary.
- S23 reuses exact source samples across its stages, in one continuous duplex
  stream. Default 1-second gaps reduce carry-over. Zero-gap simulated captures
  can fail additivity and must not be used as a passing reference fixture.
  Fixed-duration exported clips do not include the full late acoustic tail;
  the continuous raw capture is retained for subsequent resegmentation.
- The optional two-stage target-only/mixed mode retains its existing timing
  contract, but cannot measure three-stage additivity. Check quality_gate_enabled
  as well as supervision_ready if selecting only acoustically validated pairs.
- Real S4 repeated readings are explicitly strict_supervised_pair=False.
  prompt_text is a reading prompt, not verified transcription of each 4s clip.
  First-stream preparation precedes actual recording: speak only when the UI
  says it is recording, not when the prompt first appears.

## RME UCX acceptance checklist

1. Select ASIO and the same Fireface duplex device for input/output. Confirm
   48 kHz, microphone input numbers and separate mouth/noise output numbers.
2. In TotalMix, verify software playback routes to the intended physical
   outputs. For acoustic microphone recording, disable unintended loopback,
   input-to-speaker monitoring and processing that changes between stages.
3. Use conservative playback level first. Run one named S1 test with five
   repeats; inspect accepted count, reconstruction and inter-mic timing.
4. Run three S23 pairs; check output isolation, no xruns, quality_gate_passed
   and supervision_ready. Stop mid-pair and resume once before a long run.
5. Run two S4 rounds: read line 1, remain quiet, repeat line 1; then line 2.
   Confirm large text is visible on the actual laboratory display.
6. Do not change wearing, gains, routing or geometry within a paired S23 test.
   Use a new experiment name when the physical setup changes.

Hardware reference: https://rme-audio.de/downloads/fface_ucx_e.pdf
