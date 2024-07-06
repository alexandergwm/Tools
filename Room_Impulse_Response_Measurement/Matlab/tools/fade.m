function x = fade(x, Nin, Nout) %fade   Apply a raised cosine window for fade in and fade out
% 
%USAGE
%   y = fade(x, Nin, Nout)
%  
%INPUT ARGUMENTS
%         x : input signal [N x 1]
%   Nfadein : number of samples to fade in
%  Nfadeout : number of samples to fade out
% 
%OUTPUT ARGUMENTS
%   x : signal with raised cosine window applied [N x 1]



%% CHECK INPUT ARGUMENTS
%
%
% Check for proper input arguments
if nargin < 1 || nargin > 3
    help(mfilename);
    error('Wrong number of input arguments!')
end

% Set default values
if nargin < 3 || isempty(Nout); Nout = 10; end
if nargin < 2 || isempty(Nin);  Nin  = 10; end

% Check if input is a vector
if min(size(x)) > 1
    error('Input signal x must be a vector.')
else
    % Ensure x is a colum vector
    x = x(:);
end


%% WINDOWING 
% 
% 
% Dimensionality
N = numel(x);

% Reduce abrupt discontinuities at signal start and end, by fading in over
% number of samples specified by Nin and Nout.

% Create raised cosine windows for fade in and out
% -- ADD YOUR CODE HERE --------------------------------------------------
w = ones(numel(x), 1);
w(1:Nin) = 0.5-0.5*cos(pi*(1:Nin)/Nin);
w(end-Nout+1:end) = 0.5*cos(pi*(1:Nout)/Nout)+0.5;


% Apply window
x = x.*w;

