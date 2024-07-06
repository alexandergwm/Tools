%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calcEDC.m
% Written: 2019 Zach Neveu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [EDCdB, t] = calcEDC(h, fs, trunctime)
	% Returns the logarithmic energy decay for a given impulse response
	%
	% Usage: [EDCdB, t] = calcEDC(h, fs, trunctime)
	%
	% Arguments
	% h			: M channel impulse response (N samples long) [NxM]
	% fs		: Sample Rate (Hz)
	% trunctime	: Time (s) at which IR is truncated before EDC calculation
	%
	% Returns
	% EDCdB		: Logarithmic energy decay curve [tSamples x M]
	% t			: time vector [tSamples x 1]
	h = h(1:fs*trunctime);
	EDCdB = 10*log10(1-(cumsum(h.^2)/sum(h.^2)));
	t = 1:1/fs:trunctime;
end

