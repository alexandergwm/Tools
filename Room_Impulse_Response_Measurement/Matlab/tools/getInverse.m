function [hinv,Hinv] = getInverse(s)
%getInverse   Creates an inverse filter in the frequency domain.
%
%USAGE
%   [hinv,Hinv] = getInverse(s)
%  
%INPUT ARGUMENTS
%   s : single channel excitation signal [N x 1]
% 
%OUTPUT ARGUMENTS
%   hinv : impulse response of inverse filter [N x 1]
%   Hinv : complex spectrum of inverse filter [N x 1]

%   Developed with Matlab 9.2.0.538062 (R2017a). Please send bug reports to
%   
%   Author  :  Marton Marschall, © 2017
%              Technical University of Denmark
%              mmars@dtu.dk
%
%   History :
%   v.0.1   2017/08/19
%   ***********************************************************************


%% CHECK INPUT ARGUMENTS
%
%
% Check for proper input arguments
if nargin ~= 1
    help(mfilename);
    error('Wrong number of input arguments!')
end


%% CREATE INVERSE FILTER
% 
% 
% Direct inverse in the frequency domain
% -- ADD YOUR CODE HERE ----------------------------------------------
S = fft(s, numel(s));

Hinv = 1./S;

hinv = ifft(Hinv, numel(S));
