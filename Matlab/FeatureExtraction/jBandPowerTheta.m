
function BPT = jBandPowerTheta(X)
% Parameters         
f_low  = 4;      % 4 Hz
f_high = 8;      % 8 Hz

% sampling frequency 
% if isfield(opts,'fs'), fs = opts.fs; end

% Band power 
BPT = bandpower(X, 125, [f_low f_high]);
end

