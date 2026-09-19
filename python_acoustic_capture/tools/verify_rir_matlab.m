function verify_rir_matlab(root)
% Compare current Python artifacts against the installed MATLAB implementation.
rows = cell(1, 4);
for k = 1:4
    d = load(fullfile(root, sprintf('case_%d.mat', k)));
    exc = sweeptone(d.duration, d.post, d.fs, ...
        'SweepFrequencyRange', [40 22000], 'ExcitationLevel', d.level);
    est = impzest(exc, d.response);
    n = min(size(est, 1), size(d.python_ir, 1));
    r.release = version('-release');
    r.duration = d.duration;
    r.level = d.level;
    r.ess_max_error = max(abs(exc(1:numel(d.sweep)) - double(d.sweep)));
    r.ir_max_error = max(abs(est(1:n,:) - double(d.python_ir(1:n,:))), [], 'all');
    r.passed = r.ess_max_error < 2e-8 && r.ir_max_error < 1e-6;
    rows{k} = r;
end
fid = fopen(fullfile(root, 'matlab_comparison.json'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(rows, PrettyPrint=true));
disp(jsonencode(rows, PrettyPrint=true));
assert(all(cellfun(@(r) r.passed, rows)), 'Numerical comparison failed');
end
