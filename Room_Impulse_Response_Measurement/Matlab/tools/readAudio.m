function [data,fsHz,nBits] = readAudio(fName,fsRefHz,range)
%readAudio   Read audio files using either audioread or wavread.
%   Wavread is supported until R2013b and has been replaced by audioread in
%   R2014a. This function automatically detects the Matlab version used and
%   selects the proper routine for reading audio signals. 
% 
%USAGE
%   [data,fsHz,nBits] = readAudio(fName)
%   [data,fsHz,nBits] = readAudio(fName,fsRefHz,range)
%   [data,fsHz,nBits] = readAudio(fName,'info')
% 
%INPUT ARGUMENTS
%         fName : audio file name
%       fsRefHz : requested sampling frequency in Hertz, the audio signal 
%                 will be resampled accordingly (default, fsRefHz = [])
%         range : select sample range for all channels, either range = N to
%                 extract the first N samples or range = [N1 N2] to extract
%                 a specific sample range (default, range = [])
% 
%   If the second input argument is the string 'info', the function returns
%   the dimensionality and the sampling frequency of the audio file instead
%   of the signal itself. 
% 
%OUTPUT ARGUMENTS
%    data : audio signal / dimensions [nSamples x nChannels]
%    fsHz : sampling frequency in Hertz
%   nBits : number of bits per sample

%   Developed with Matlab 8.5.0.197613 (R2015a). Please send bug reports to
%   
%   Author  :  Tobias May, © 2015-2016
%              Technical University of Denmark
%              tobmay@elektro.dtu.dk
%
%   History :
%   v.0.1   2015/10/08
%   v.0.2   2016/02/23 added nBits to output arguments
%   ***********************************************************************


%% CHECK INPUT ARGUMENTS 
% 
% 
% Check for proper input arguments
if nargin < 1 || nargin > 3
    help(mfilename);
    error('Wrong number of input arguments!')
end

% Set default values
if nargin < 2 || isempty(fsRefHz); fsRefHz = []; end
if nargin < 3 || isempty(range);   range   = []; end

% Check validity of second input argument
if ~isempty(fsRefHz) && ~ischar(fsRefHz) && ~isnumeric(fsRefHz)
    error('Second input must be either a string or a numeric array.')
end

% Check validity of third input argument
if ~isempty(range) && (~isnumeric(range) || numel(range) > 2)
    error('Third input must be a 1- or 2-dimensional numeric array.')
end


%% IDENTIFY IF WAVREAD IS AVAILABLE
% 
% 
% Use cache 
persistent PERbUseWAV

% Check if wavread is available
if isempty(PERbUseWAV)
    
    % Get Matlab version
    strVersion = version('-release');
    
    if str2double(strVersion(1:4)) < 2013
        PERbUseWAV = true;
    else
        PERbUseWAV = false;
    end
end


%% DETECT IF FILE INFO OR THE DATA SHOULD BE RETURNED
% 
% 
% If second input arguement is a character array
if ~isempty(fsRefHz) && ischar(fsRefHz)
    if strcmpi(fsRefHz,'info')
        % Return signal info
        bInfo = true;
    else
        error('Input argument ''%s'' is not supported!',fsRefHz)
    end
else
    % Return signal
    bInfo = false;
end


%% READ AUDIO SIGNAL
% 
% 
% Check if file info or data was requested
if bInfo
    % Read file dimensions either using wavread or audioread
    if PERbUseWAV
        [data,fsHz,nBits] = wavread(fName,'size'); %#ok 
    else
        finfo = audioinfo(fName);
        data  = [finfo.TotalSamples finfo.NumChannels];
        fsHz  = finfo.SampleRate;
        nBits = finfo.BitsPerSample;
    end    
else
    % Ensure range is two-dimensional for compatibility with audioread
    if ~isempty(range) && numel(range) == 1
        range = [1 range];
    end
    
    % Read file either using wavread or audioread
    if PERbUseWAV
        if isempty(range)
            [data,fsHz,nBits] = wavread(fName); %#ok
        else
            [data,fsHz,nBits] = wavread(fName,range); %#ok
        end
    else
        if isempty(range)
            [data,fsHz] = audioread(fName);
        else
            [data,fsHz] = audioread(fName,range); 
        end
        if nargout > 2
           % Read out nBits
           finfo = audioinfo(fName);
           nBits = finfo.BitsPerSample;
        end
    end
end


%% RESAMPLE DATA, IF REQUIRED
% 
% 
% Check if sampling frequencies differ
if ~bInfo && ~isempty(fsRefHz) && ~isequal(fsHz,fsRefHz)
    % Resample data
    data = resample(data,fsRefHz,fsHz);
    
    % Update sampling frequency
    fsHz = fsRefHz;
end
