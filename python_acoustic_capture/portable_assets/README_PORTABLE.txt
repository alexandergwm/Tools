Acoustic Capture Portable
=========================

1. Double-click AcousticCapture.exe or AcousticCapturePortable.exe.
2. No Python installation is required.
3. The RME Windows driver must still be installed for Fireface/ASIO access.
4. On first launch, editable configs and demo audio are created next to the EXE.
5. Captures are saved under the runs folder next to the EXE.
6. Dataset exports can be saved under the datasets folder.
7. For checklist-driven work, open acoustic_capture_checklist_template.xlsx,
   add one physical experiment per row, then use the GUI's XLSX checklist
   button. The app loads the selected name/labels and writes completion status
   back to the workbook. Close the workbook in Excel before capture completes
   so Windows does not lock it.

Manual naming remains available when no checklist row is selected.

For supervised speech data, connect microphone input, artificial-mouth output,
and interferer output to the same RME interface and select the same RME ASIO
device for both recording and playback.

The executable is not code-signed. Windows SmartScreen may show a warning on a
new computer; use More info > Run anyway only when the file came from your own
trusted build or repository release.
