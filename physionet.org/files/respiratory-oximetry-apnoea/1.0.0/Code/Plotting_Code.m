%Plotting code for pulse data and inline sensor data
%Jordan Hil
%11/06/2024

clc
clear 
close all

%% Load Data ------------------------------------------------------------
subject_num = 1;                       % subject number 
trial_number = 4;                       % trial number (1-7 with inline sensor) 
Fs = 250;                               % sampling frequency for pulse oximter

%% Load pulse oximeter data
for j = subject_num 
    for i = trial_number          
        subject{j}.test{i}.data_path = sprintf("Neck_Pulse_Oximeter_Data_timefixed/Subject_%d/*.csv", subject_num);      %path for all data in subject folder
        subject{j}.test{i}.data_files = dir(subject{j}.test{i}.data_path);
        subject{j}.test{i}.num_files = length(subject{j}.test{i}.data_files);

        for filenum = 1:subject{j}.test{i}.num_files 
            subject{j}.test{i}.filenames{filenum} = subject{j}.test{i}.data_files(filenum).name;
        end
        subject{j}.test{i}.sorted_datanames = natsort(string(subject{j}.test{i}.filenames)); % Sort the files chronologically

        for filenum = 1:subject{j}.test{i}.num_files
            temp_data_path = sprintf("Neck_Pulse_Oximeter_Data_timefixed/Subject_%d/%s", subject_num, subject{j}.test{i}.sorted_datanames(trial_number)); % Get the data path 
            subject{j}.test{i}.loaded_data = readtable(temp_data_path, 'HeaderLines', 1); % load the data and skip first line
        end 
    end
end



%% Load Inline Sensor data
for j = subject_num                           
    for i = trial_number         
        subject{j}.test{i}.data_path_breath = sprintf("Inline_PQ_Data/Subject%d/*.csv", subject_num);      %path for all data in subject folder
        subject{j}.test{i}.data_files_breath = dir(subject{j}.test{i}.data_path_breath);
        subject{j}.test{i}.num_files_breath = length(subject{j}.test{i}.data_files_breath);

        for filenum = 1:subject{j}.test{i}.num_files_breath 
            subject{j}.test{i}.filenames_breath{filenum} = subject{j}.test{i}.data_files_breath(filenum).name;
        end
        subject{j}.test{i}.sorted_datanames_breath = natsort(string(subject{j}.test{i}.filenames_breath)); % Sort the files chronologically

        for filenum = 1:subject{j}.test{i}.num_files_breath
            temp_data_path = sprintf("Inline_PQ_Data/Subject%d/%s", subject_num, subject{j}.test{i}.sorted_datanames_breath(trial_number)); % Get the data path 
            subject{j}.test{i}.loaded_data_breath = readtable(temp_data_path, 'HeaderLines', 1); % load the data and skip first line
            % order - Time [s]	,Gauge Pressure [cmH2O],	Inspiratory differenrtial pressure [cmH2O],	Start Time
        end 
    end
end



%% Filters Pulse Data --------------------------------------------------------------
% All frequency values are in Hz

% Filter for pulse data
N  = 6;         % Order
Fc = 8;         % Cutoff Frequency
f_da  = fdesign.lowpass('N,F3dB', N, Fc, Fs);
Fda = design(f_da, 'butter');        

%Filter the data
for j = subject_num
    for i = trial_number
        for k = 2:9      %as k(1) is time 
            temp_data = -subject{j}.test{i}.loaded_data{:,k};       %negative due to photodiode 

            %filter the data with filtfilt for zerophase shift 
            subject{j}.test{i}.unfiltered{k-1} = temp_data;
            subject{j}.test{i}.filtered{k-1} = filtfilt(Fda.sosMatrix,Fda.ScaleValues,temp_data);
        end
    end
end

%% Remove first and last 2 seconds of filtered data -----
% this accounts for the filter rise and end period when data has not been
% filtered bang on

for j = subject_num
    for i = trial_number
        for k = 2:9
            % Define the number of samples to remove
            samples_to_remove = 500;
            
            % Process filtered signals
            fields = {'filtered'};
            for field = fields
                temp_data = subject{j}.test{i}.(field{1}){k-1};
                
                % Ensure the data length is sufficient
                if length(temp_data) > 2 * samples_to_remove
                    % Remove first and last 500 samples
                    processed_data = temp_data(samples_to_remove:end - samples_to_remove);
                    subject{j}.test{i}.(field{1}){k-1} = processed_data;
                   
                end
            end
        end
    end
end


%% Time --------------------------------------------------------------
time_test = cell(trial_number, 1);
time_unchanged_test = cell(trial_number, 1);

for i = trial_number          % Test
    %Adjust from date timing from data to seconds
    time_unchanged = subject{j}.test{i}.loaded_data{:,1};
    time = time_unchanged(1:length(time_unchanged)-999); %as remove first and last 500 samples above

    time_test{i} = time; 
    time_unchanged_test{i} = time_unchanged;
end



%% Start time for the breathing data

% Extract data using Var1–Var3
breath_data = subject{j}.test{i}.loaded_data_breath;
time_breath = breath_data.Var1(1:19999);
gauge_pressure = breath_data.Var2(1:19999);
diff_pressure = breath_data.Var3(1:19999);

% Extract start datenum from the first row of column 4
start_datenum = subject{j}.test{i}.loaded_data_breath{1,4};
% Convert datenum to datetime
start_datetime = datetime(start_datenum, 'ConvertFrom', 'datenum', 'Format', 'dd-MMM-yyyy HH:mm:ss.SSSSSS');



%% Plotting ------------------------------------------------------------
for j = subject_num
    for i = trial_number     

        % Unfiltered pulse data
        figure(10+i)
        sgtitle('Unfiltered Pulse Data')
        titles = {'PD1 660nm', 'PD2 660nm', 'PD1 940nm', 'PD2 940nm', 'PD3 660nm', 'PD4 660nm', 'PD3 940nm', 'PD4 940nm'};
        index_adjustments = [0, 0, 2, 2, -2, -2, 0, 0];
        
        for ii = 1:8
            subplot(4,2,ii)
            plot(time_unchanged_test{i}, subject{j}.test{i}.unfiltered{ii + index_adjustments(ii)}, LineWidth=1.5)
        
            if ~isempty(titles{ii})
                title(titles{ii})
            end
            grid on
            ylabel('Voltage (V)')
            xlabel('Time (s)')
        end


        %plot filtered Pulse Data
        figure(20+i)
        sgtitle('Filtered Pulse Data')
        titles = {'PD1 660nm', 'PD2 660nm', 'PD1 940nm', 'PD2 940nm', 'PD3 660nm', 'PD4 660nm', 'PD3 940nm', 'PD4 940nm'};
        index_adjustments = [0, 0, 2, 2, -2, -2, 0, 0];
        
        for ii = 1:8
            subplot(4,2,ii)
            plot(time_test{i}, subject{j}.test{i}.filtered{ii + index_adjustments(ii)}, LineWidth=1.5)
        
            if ~isempty(titles{ii})
                title(titles{ii})
            end
            grid on
            ylabel('Voltage (V)')
            xlabel('Time (s)')
        end

        % Gauge and differential pressure data
        figure(30+i)
        sgtitle('Inline Sensor Data')
        subplot(2,1,1)
        plot(time_breath, gauge_pressure, LineWidth=1.5)
        grid on
        ylabel('Gauge Pressure (cmH2O)')
        xlabel('Time (s)')

        subplot(2,1,2)
        plot(time_breath, diff_pressure, LineWidth=1.5)
        grid on
        ylabel('Differential Pressure (cmH2O)')
        xlabel('Time (s)')
        xlim([0, 50])
       
    end
end

%% plot pressure & pulse on same plot --------------------------------------------------

figure(50)
%sgtitle('Pulse and Pressure Data')

titles = {'PD1 660nm', 'PD2 660nm', 'PD1 940nm', 'PD2 940nm', ...
          'PD3 660nm', 'PD4 660nm', 'PD3 940nm', 'PD4 940nm', ...
          'Gauge Pressure', 'Differential Pressure'};

index_adjustments = [0, 0, 2, 2, -2, -2, 0, 0];

% Plot the 8 filtered signals
for ii = 1:8
    subplot(5, 2, ii)
    plot(time_test{i}, subject{j}.test{i}.filtered{ii + index_adjustments(ii)}, 'LineWidth', 1.5)
    title(titles{ii})
    grid on
    ylabel('Voltage (V)')
    xlabel('Time (s)')
end


% Plot gauge pressure
subplot(5, 2, 9)
plot(time_breath, gauge_pressure, 'b', 'LineWidth', 1.5)
title(titles{9})
grid on
ylabel('Pressure (cmH2O)')
xlabel('Time (s)')
xlim([0, 50])

% Plot differential pressure
subplot(5, 2, 10)
plot(time_breath, diff_pressure, 'r', 'LineWidth', 1.5)
title(titles{10})
grid on
ylabel('Pressure (cmH2O)')
xlabel('Time (s)')
xlim([0, 50])


% differential pressure and pules
% Plot both on the same graph with different axes using yyaxis
figure(60);
t = tiledlayout(4, 1); % Create a 4x1 tiled layout

titles = {'PD1', 'PD2', 'PD3', 'PD4'};


for ii = 1:4
    nexttile; % Move to the next subplot

    % Plot using yyaxis for dual y-axis
    yyaxis left;
    plot(time_test{i}, subject{j}.test{i}.filtered{ii}, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); % Blue
    ax = gca;
    ax.YColor = [0 0.4470 0.7410]; % Match y-axis color with the line

    yyaxis right;
    plot(time_breath, diff_pressure, 'Color', [0.6350 0.0780 0.1840], 'LineWidth', 2); % Red for diff pressure
    ax.YColor = [0.6350 0.0780 0.1840];

    % Title and grid settings
    if ~isempty(titles{ii})
        title(titles{ii}, 'FontName', 'Times New Roman', 'FontSize', 16);
    end
    grid on;
    grid minor;

    % Set font properties
    ax.FontName = 'Times New Roman';
    ax.FontSize = 16;

end

% Shared x-axis label
xlabel(t, 'Time (s)', 'FontName', 'Times New Roman', 'FontSize', 20, 'FontWeight', 'bold');
title(t, 'Filtered Pulse Signals and Differential Pressure', ...
    'FontName', 'Times New Roman', 'FontSize', 22, 'FontWeight', 'bold');


% Gauge presure and Pulse
% Plot both on the same graph with different axes using yyaxis
figure(70);
t = tiledlayout(4, 1); % Create a 4x1 tiled layout

titles = {'PD1', 'PD2', 'PD3', 'PD4'};

for ii = 1:4
    nexttile; % Move to the next subplot

    % Plot using yyaxis for dual y-axis
    yyaxis left;
    plot(time_test{i}, subject{j}.test{i}.filtered{ii}, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); % Blue
    ax = gca;
    ax.YColor = [0 0.4470 0.7410]; % Match y-axis color with the line

    yyaxis right;
    plot(time_breath, gauge_pressure, 'Color', [0.6350 0.0780 0.1840], 'LineWidth', 2); % Red for diff pressure
    ax.YColor = [0.6350 0.0780 0.1840];

    % Title and grid settings
    if ~isempty(titles{ii})
        title(titles{ii}, 'FontName', 'Times New Roman', 'FontSize', 16);
    end
    grid on;
    grid minor;

    % Set font properties
    ax.FontName = 'Times New Roman';
    ax.FontSize = 16;

    xlim([0, 50])
end

% Shared x-axis label
xlabel(t, 'Time (s)', 'FontName', 'Times New Roman', 'FontSize', 20, 'FontWeight', 'bold');
title(t, 'Filtered Pulse Signals and Gauge Pressure', ...
    'FontName', 'Times New Roman', 'FontSize', 22, 'FontWeight', 'bold');




%% new plot
figure(860 + i); clf
t = tiledlayout(6,1); % 6 rows, 1 column
t.TileSpacing = 'compact';
t.Padding = 'compact';

titles = {'PD1', 'PD2', 'PD3', 'PD4', 'Gauge Pressure', 'Differential Pressure'};

% --- First 4 tiles: PD pairs (660nm vs 940nm) ---
for ii = 1:4
    nexttile;
    
    yyaxis left;
    plot(time_test{i}, subject{j}.test{i}.filtered{ii}, ...
        'Color', [0.6350 0.0780 0.1840], 'LineWidth', 2); % Dark red for 660nm
    ax = gca;
    ax.YColor = [0.6350 0.0780 0.1840];
    if ii == 1
        ylabel('660nm (V)', 'FontName','Times New Roman', 'FontSize',14)
    else
        ylabel('')
    end

    yyaxis right;
    plot(time_test{i}, subject{j}.test{i}.filtered{ii+4}, ...
        'Color', [0 0.4470 0.7410], 'LineWidth', 2); % Blue for 940nm
    ax.YColor = [0 0.4470 0.7410];
    if ii == 1
        ylabel('940nm (V)', 'FontName','Times New Roman', 'FontSize',14)
    else
        ylabel('')
    end

    title(titles{ii}, 'FontName','Times New Roman', 'FontSize',16)
    grid on; grid minor;
    xlim([0 50])
    
    if ii < 7
        set(gca,'XTickLabel',[]) % hide x labels except bottom
    end
end

% --- 5th tile: Gauge pressure ---
nexttile
plot(time_breath, gauge_pressure, 'b', 'LineWidth', 2)
title(titles{5}, 'FontName','Times New Roman', 'FontSize',16)
ylabel('cmH_2O', 'FontName','Times New Roman', 'FontSize',14)
grid on; grid minor;
xlim([0 50])
set(gca,'XTickLabel',[])

% --- 6th tile: Differential pressure ---
nexttile
plot(time_breath, diff_pressure, 'r', 'LineWidth', 2)
title(titles{6}, 'FontName','Times New Roman', 'FontSize',16)
ylabel('cmH_2O', 'FontName','Times New Roman', 'FontSize',14)
grid on; grid minor;
xlim([0 50])

% Shared x-axis label at bottom
xlabel(t, 'Time (s)', 'FontName','Times New Roman', ...
    'FontSize',20, 'FontWeight','bold')
