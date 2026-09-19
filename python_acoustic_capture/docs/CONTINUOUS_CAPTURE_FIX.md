# Continuous capture repair — 2026-09-19

## Implemented

- Each RIR, speech-pair, human-speech or record-only experiment uses a lazy,
  thread-affine input/duplex stream kept open until the workflow ends.
- Before measuring, discard idle input and output silence until callbacks have
  been continuous for one second. Confirm readiness again at the callback that
  actually starts the job. The UI shows preparation, not recording, during this
  interval. Idle audio is not saved.
- Record wall-clock callback gaps separately from PortAudio xrun flags. Gaps
  greater than max(150 ms, four callback periods) mark timing_discontinuity.
  This is a conservative continuity warning, not an exact dropped-sample count.
- RIR and speech quality checks reject flagged measurements. Basic recording
  reports a warning. Human-stage labels include continuity/quality fields.
- Cancelled RIR takes are saved as partial and excluded from the mean.
- Cancelled speech pairs save partial audio, an invalid label, and an immutable
  attempt entry. Their resume checkpoint is not advanced; resume re-records the
  unfinished pair. Raw attempts remain archived across resume.
- Automatic speech faults default valid to no. GUI completion warnings include
  continuity failures and unapproved supervision pairs.
- Preserve common multichannel sample indices: do not independently align each
  microphone, split input/output into falsely synchronized streams, or lower
  reconstruction thresholds to hide bad captures.

## Verified

- 154 automated tests passed, including new readiness, silent-gap detection,
  cleanup, partial-label/resume and GUI warning checks.
- Real MME playback/recording was exercised repeatedly with the user's USB
  Beosound, one input channel, including complete and stopped runs of all four
  workflows. Synthetic low-level test material was used, not a human corpus.
- In the final hardware round, all four stop paths retained partial audio.
  The stopped speech pair had one invalid label row and could be resumed in
  the regression test.
- Speech playback references reuse exactly the same target/interferer samples
  in isolated and mixed segments. Bad transport remains invalid regardless.

## NOT certified / remaining hardware blocker

- Beosound input still intermittently stalls. A standalone native WinMM test,
  without sounddevice, PortAudio or this application, reproduced a 3.310 s gap.
  MME and WASAPI input probes also reproduced it. This locates the issue below
  the application; it does not uniquely identify the Windows driver, processing
  layer, USB path or device firmware.
- Stable startup improves subsequent recording but does not guarantee that
  this device never stalls later. Code now flags later gaps instead of treating
  a full-length WAV as proof of continuous capture.
- Final RIR run produced an accepted individual take and an average artifact,
  while rejecting the other take. One accepted take is not repeatability proof.
- Final S23 run still had a timing fault and was correctly marked valid=no,
  supervision_ready=no. It is NOT a certified supervised training pair.
- ASIO/RME, physical multichannel delay, independent target/noise speakers,
  actual participants and full real-hardware GUI-click validation remain untested.
- The first preparation can still take several seconds. This is NOT a claim
  that a cold device immediately records continuously after a click. Achieving
  that requires a separate, explicit device-preparation phase before Start.
- Portable EXE has not been rebuilt; restart the source-based application to
  load these changes. No system settings or original user YAML were changed.

Local diagnostic evidence is under outputs/mme_audit_20260919/, especially
four_mode_final/audit.json, four_mode_final/artifact_checks.json and
native_wavein_report.json. Real recordings are excluded from Git.

Next acceptance step: use a separate wired microphone or the lab RME, repeat
the same four workflows, and verify continuity, RIR reconstruction and paired
supervision quality without relaxing thresholds.
