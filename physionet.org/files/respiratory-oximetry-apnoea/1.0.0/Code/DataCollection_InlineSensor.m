%% Data Collection for A1 PrototypeA
% Ella Guy 
% Last Updated: 10NOV2024

% Data from arduino serial to MATLAB
% Scales data into pressure [Pa]
% Saves outfile as a .mat file 

clear
clc
close all

%% Initialisation

% Trial Information =======================================================
timeLength = 10; % recording length of the trial
SamplingFrequency = 500; %[Hz]

% Inputs-------------------------------------------------------------------
PEEP = 0;  % PEEP [cmH2O]
C = 0;   % Compliance [L/cmH2O]
R = 0;     % Parabolic Resistance 
V = 0;    % Tidal Volume [mL]
%------------------------------------------\-------------------------------

%==========================================================================
% delete(instrfindall);
% 
% Arduino Communication Information =======================================
comPort = '/dev/cu.usbserial-1130';  % Check COM port "serialportlist()"
baudRate = 115200; % Match with arduino baudrate
%==========================================================================

%Predefining variables to be read from arduino ----------------------------
dataLength = SamplingFrequency*timeLength;

GaugeP_raw = []; %zeros(dataLength, 1);             
InhaleP_raw = []; %zeros(dataLength, 1);          

t = []; %zeros(dataLength, 1);  % Time using computer clock
%--------------------------------------------------------------------------

% Serial open
arduino = serialport(comPort, baudRate);

%% Read data from arduino to MATLAB arrays

% Open comms to arduino and initialise clock and index
t0 = clock;
i = 1;
StartTime = datetime;

% Collecting data for the defined time 
while etime(clock, t0) < timeLength
    y = fscanf(arduino, '%s');
    out = sscanf(y, '%f,%f');

    if length(out) == 2 
    t(i) = etime(clock, t0);      
    InhaleP_raw(i) = out(1);   
    GaugeP_raw(i) = out(2);
    i = i+1;  
    end 
end

% Close the connection with Arduino
clear arduino;

%% Data Unit Conversions

% Initising time array
time = t - t(1);

P_max = 1; % [psi]
P_min = -1; % [psi]

% Venturi Differential Pressures ------------------------------------------
GaugeP = GaugeP_raw; % [cmH2O]
InhaleDeltaP = InhaleP_raw; %[cmH2O] 
%--------------------------------------------------------------------------

%% Plots of collected data (with converted units)

figure(1) %----------------------------------------------------------------
subplot(3, 1, 1)
plot(time, GaugeP, '.')
grid on
grid minor
title("Gauge Pressure [cmH_2O]", 'Fontsize', 12)

subplot(3, 1, 2)
plot(time, InhaleDeltaP, '.')
grid on
grid minor
title("Differential Venturi Pressure (Inhalatory Direction)[cmH_2O]", 'Fontsize', 12)

%% Generates Outfiles 

% Files with unit conversions ---------------------------------------------
outfile_format = 'AireA1_PrototypeA_MechanicalLung_P%.1f_C%.2f_R%d_V%dml.mat';
outfile = sprintf(outfile_format, PEEP, C, R, V);

save(outfile, 'time', 'GaugeP', 'InhaleDeltaP', 'StartTime');

%--------------------------------------------------------------------------