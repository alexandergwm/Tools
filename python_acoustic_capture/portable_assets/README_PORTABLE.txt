Acoustic Capture Portable
=========================

1. Double-click AcousticCapture.exe or AcousticCapturePortable.exe.
2. No Python installation is required.
3. The RME Windows driver must still be installed for Fireface/ASIO access.
4. On first launch, editable configs and demo audio are created next to the EXE.
   Each release uses a versioned default config, so upgrading does not overwrite
   a config you edited with an older release.
5. Captures are saved under the runs folder next to the EXE.
6. Dataset exports can be saved under the datasets folder.
7. For checklist-driven work, open acoustic_capture_checklist_template.xlsx,
   add one physical experiment per row, then use the GUI's XLSX checklist
   button. The app loads the selected name/labels and writes completion status
   back to the workbook. Close the workbook in Excel before capture completes
   so Windows does not lock it.

Manual naming remains available when no checklist row is selected.

RIR capture defaults to automatic TrajectoRIR-inspired selection. Choose the
fixed-count option in the RIR panel when you need an exact number of raw takes
for later offline reselection; fixed-count capture never stops early on
convergence, and the automatic RIR is kept only as a reference result.

Large speech batches are checkpointed after every completed source pair. If a
run is stopped or fails, enter the same experiment name next time and accept
the resume prompt. Edited labels.xlsx files can be imported from the File menu;
dataset compilation then prefers labels_reviewed.jsonl.

For supervised speech data, connect microphone input, artificial-mouth output,
and interferer output to the same RME interface and select the same RME ASIO
device for both recording and playback.

The executable is not code-signed. Windows SmartScreen may show a warning on a
new computer; use More info > Run anyway only when the file came from your own
trusted build or repository release.
