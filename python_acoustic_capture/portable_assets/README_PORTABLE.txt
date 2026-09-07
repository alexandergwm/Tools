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

For laboratory ASIO capture, prefer this extracted folder build. It runs as one
application process and keeps the driver DLLs and logs in inspectable paths.
The single-file EXE remains available for ordinary Windows audio trials.

Manual naming remains available when no checklist row is selected.

RIR capture defaults to an exact take count. Every take is deconvolved, then
reconvolved with the original ESS and compared with the corresponding real
microphone sweep using MSE, NMSE, and correlation. The final RIR is the aligned
arithmetic mean of all takes accepted by recording QC. A second mode keeps the
same raw files but marks the generated mean as a reference for later offline
reselection.

Large speech batches are checkpointed after every completed source pair. If a
run is stopped or fails, enter the same experiment name next time and accept
the resume prompt. Edited labels.xlsx files can be imported from the File menu;
dataset compilation then prefers labels_reviewed.jsonl.

For supervised speech data, connect microphone input, artificial-mouth output,
and interferer output to the same RME interface and select the same RME ASIO
device for both recording and playback.

ASIO safety: stream open/start/stop/close operations stay on one audio-owner
thread. The GUI only posts a stop request. Monitoring is closed before capture
and disabled while the ASIO device is in use.

The executable is not code-signed. Windows SmartScreen may show a warning on a
new computer; use More info > Run anyway only when the file came from your own
trusted build or repository release.
