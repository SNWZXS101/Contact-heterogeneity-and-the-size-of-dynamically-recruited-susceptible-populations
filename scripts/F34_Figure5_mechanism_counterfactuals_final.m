%% F34_Figure5_mechanism_counterfactuals_final_v2.m
% Final main-text Figure 5, simplified three-panel design.
%
% Core message:
%   Mechanisms that suppress incidence differ in how they reduce
%   high-risk susceptible headcount and edge-weighted burden.
%
% Panels:
%   A  Incidence reduction: median and IQR across seven representative waves
%   B  Headcount versus edge-weighted burden: median and IQR
%   C  Mechanism map in headcount-edge space
%
% Input:
%   outputs/results/policy_counterfactual_summary_final.csv
%
% Outputs:
%   outputs/figures/Figure5_mechanism_counterfactuals.fig
%   outputs/figures/Figure5_mechanism_counterfactuals.png
%   outputs/figures/Figure5_mechanism_counterfactuals.jpg
%   outputs/figures/Figure5_mechanism_counterfactuals.pdf
%   outputs/figures/Figure5_mechanism_counterfactuals.svg
%   outputs/figures/source_data/Figure5_source_data.csv
%   outputs/tables/main_table3_counterfactuals_final.csv
%   outputs/tables/main_table3_counterfactuals_final.tex
%
% ASCII-only source for broad MATLAB compatibility.
% Prose uses Arial; mathematical labels use Times New Roman.

clear; close all; clc;

SCRIPT_DIR = fileparts(mfilename('fullpath'));
ROOT = fileparts(SCRIPT_DIR);

RESULTS = fullfile(ROOT,'outputs','results');
FIG_DIR = fullfile(ROOT,'outputs','figures');
SRC_DIR = fullfile(FIG_DIR,'source_data');
TABLE_DIR = fullfile(ROOT,'outputs','tables');

if ~exist(FIG_DIR,'dir'), mkdir(FIG_DIR); end
if ~exist(SRC_DIR,'dir'), mkdir(SRC_DIR); end
if ~exist(TABLE_DIR,'dir'), mkdir(TABLE_DIR); end

inputFile = fullfile(RESULTS,'policy_counterfactual_summary_final.csv');
D = readtable(inputFile);

requiredVars = { ...
    'wave_id','country','scenario', ...
    'percent_reduction_peak_incidence', ...
    'percent_reduction_cumulative_incidence', ...
    'percent_reduction_max_S_head', ...
    'percent_reduction_max_S_edge'};

for i = 1:numel(requiredVars)
    if ~ismember(requiredVars{i},D.Properties.VariableNames)
        error('Missing required column: %s',requiredVars{i});
    end
end

% One mechanism order for every panel and Table 3.
scenarioLong = { ...
    'Reservoir gating (a -50%)', ...
    'Broad contact reduction (beta -30%)', ...
    'Faster detection/isolation (gamma +50%)', ...
    'Immune protection (25% susceptible protected)', ...
    'Target highest-activity quartile (top 3 classes -50%)', ...
    'Adaptive combined package'};

scenarioShort = { ...
    'Reservoir gating', ...
    'Broad contact reduction', ...
    'Faster isolation', ...
    'Immune protection', ...
    'High-activity targeting', ...
    'Combined package'};

scenarioTable = { ...
    '$a\rightarrow0.5a$', ...
    '$\beta_0\rightarrow0.7\beta_0$', ...
    '$\gamma\rightarrow1.5\gamma$', ...
    '$(U_j,S_j)\rightarrow0.75(U_j,S_j)$', ...
    'top three of 12 classes: $z_j\rightarrow0.5z_j$', ...
    '$a\rightarrow0.5a$, $\beta_0\rightarrow0.75\beta_0$, $\gamma\rightarrow1.5\gamma$, top-class $z_j\rightarrow0.5z_j$, $(U_j,S_j)\rightarrow0.85(U_j,S_j)$'};

scenarioCol = localCellstr(D.scenario);

K = numel(scenarioLong);
rowsByScenario = cell(K,1);

for i = 1:K
    rowsByScenario{i} = D(strcmp(scenarioCol,scenarioLong{i}),:);
    if isempty(rowsByScenario{i})
        error('Scenario not found in final CSV: %s',scenarioLong{i});
    end
end

% Palette
BLUE   = [0.00 0.45 0.70];
ORANGE = [0.84 0.37 0.00];
GREEN  = [0.00 0.62 0.45];
GRAY   = [0.50 0.50 0.50];
LIGHTGRAY = [0.82 0.82 0.82];
TARGET_BG = [1.00 0.95 0.87];

% Mechanism colors for Panel C.
MECH_COL = [ ...
    0.40 0.52 0.68;  % gating
    0.34 0.68 0.82;  % contact
    0.52 0.44 0.68;  % isolation
    0.00 0.62 0.45;  % immunity
    0.84 0.37 0.00;  % targeting
    0.70 0.20 0.20]; % combined

metricVars = { ...
    'percent_reduction_peak_incidence', ...
    'percent_reduction_cumulative_incidence', ...
    'percent_reduction_max_S_head', ...
    'percent_reduction_max_S_edge'};

med = zeros(K,4);
q25 = zeros(K,4);
q75 = zeros(K,4);

for i = 1:K
    T = rowsByScenario{i};
    for j = 1:4
        x = double(T.(metricVars{j}));
        x = x(isfinite(x));
        q = localQuantile(x,[0.25 0.50 0.75]);
        q25(i,j) = q(1);
        med(i,j) = q(2);
        q75(i,j) = q(3);
    end
end

targetIndex = 5;
immuneIndex = 4;

%% ------------------------------------------------------------------------
% Figure canvas and manually controlled axes positions.
% -------------------------------------------------------------------------
fig = figure('Color','w','Units','inches','Position',[0.25 0.25 13.6 7.1]);

axA = axes('Parent',fig,'Position',[0.08 0.13 0.28 0.76]);
axB = axes('Parent',fig,'Position',[0.405 0.13 0.29 0.76]);
axC = axes('Parent',fig,'Position',[0.755 0.18 0.22 0.66]);

annotation(fig,'textbox',[0.04 0.945 0.92 0.04], ...
    'String','Mechanisms that suppress incidence differ in how they reduce susceptible headcount and edge-weighted burden', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontWeight','bold', ...
    'FontSize',14);

%% ------------------------------------------------------------------------
% Panel A. Incidence reduction: median and IQR only
% -------------------------------------------------------------------------
hold(axA,'on');

ypos = K:-1:1;
off = 0.14;

for i = 1:K
    y = ypos(i);

    % Peak incidence
    plot(axA,[q25(i,1) q75(i,1)],[y-off y-off],'-', ...
        'Color',BLUE,'LineWidth',2.5);
    plot(axA,med(i,1),y-off,'o', ...
        'MarkerFaceColor',BLUE,'MarkerEdgeColor','w','MarkerSize',7.5);

    % Cumulative incidence
    plot(axA,[q25(i,2) q75(i,2)],[y+off y+off],'-', ...
        'Color',ORANGE,'LineWidth',2.5);
    plot(axA,med(i,2),y+off,'s', ...
        'MarkerFaceColor',ORANGE,'MarkerEdgeColor','w','MarkerSize',7.0);
end

plot(axA,[0 0],[0.4 K+0.6],'--','Color',GRAY,'LineWidth',0.8);

xlim(axA,[-5 105]);
ylim(axA,[0.4 K+0.6]);

set(axA, ...
    'YTick',1:K, ...
    'YTickLabel',fliplr(scenarioShort), ...
    'FontName','Arial', ...
    'FontSize',8.2, ...
    'Box','off');

xlabel(axA,'reduction relative to frozen M2 baseline (%)','FontName','Arial');
title(axA,'A  Incidence suppression', ...
    'HorizontalAlignment','left','FontName','Arial','FontWeight','bold');

grid(axA,'on');
set(axA,'GridAlpha',0.12);
axA.YGrid = 'off';

hPeak = plot(axA,nan,nan,'o', ...
    'MarkerFaceColor',BLUE,'MarkerEdgeColor','w','MarkerSize',7);
hCum = plot(axA,nan,nan,'s', ...
    'MarkerFaceColor',ORANGE,'MarkerEdgeColor','w','MarkerSize',6.7);

legend(axA,[hPeak hCum],{'peak incidence','cumulative incidence'}, ...
    'Location','southeast', ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',7.6);

% Small note
text(axA,0.02,0.02,'points = medians; lines = IQR across seven representative waves', ...
    'Units','normalized','FontName','Arial','FontSize',6.8, ...
    'Color',[0.35 0.35 0.35]);

%% ------------------------------------------------------------------------
% Panel B. Headcount versus edge-weighted burden
% -------------------------------------------------------------------------
hold(axB,'on');

yTarget = ypos(targetIndex);

patch(axB,[-12 105 105 -12], ...
    [yTarget-0.38 yTarget-0.38 yTarget+0.38 yTarget+0.38], ...
    TARGET_BG,'EdgeColor','none','FaceAlpha',0.85);

for i = 1:K
    y = ypos(i);

    % Median connector
    plot(axB,[med(i,3) med(i,4)],[y y],'-', ...
        'Color',[0.42 0.42 0.42],'LineWidth',2.1);

    % IQR
    plot(axB,[q25(i,3) q75(i,3)],[y-0.10 y-0.10],'-', ...
        'Color',BLUE,'LineWidth',2.2);
    plot(axB,[q25(i,4) q75(i,4)],[y+0.10 y+0.10],'-', ...
        'Color',ORANGE,'LineWidth',2.2);

    % Medians
    plot(axB,med(i,3),y,'o', ...
        'MarkerFaceColor',BLUE,'MarkerEdgeColor','w','MarkerSize',7.5);
    plot(axB,med(i,4),y,'s', ...
        'MarkerFaceColor',ORANGE,'MarkerEdgeColor','w','MarkerSize',7.0);
end

plot(axB,[0 0],[0.4 K+0.6],'--','Color',GRAY,'LineWidth',0.8);

% Direct labels for the headline targeting contrast.
text(axB,med(targetIndex,3)-2.5,yTarget-0.22, ...
    sprintf('%.1f%%',med(targetIndex,3)), ...
    'FontName','Arial','FontSize',8,'FontWeight','bold', ...
    'Color',BLUE,'HorizontalAlignment','right');

text(axB,med(targetIndex,4)+2.5,yTarget+0.22, ...
    sprintf('%.1f%%',med(targetIndex,4)), ...
    'FontName','Arial','FontSize',8,'FontWeight','bold', ...
    'Color',ORANGE,'HorizontalAlignment','left');

xlim(axB,[-12 82]);
ylim(axB,[0.4 K+0.6]);

set(axB, ...
    'YTick',1:K, ...
    'YTickLabel',fliplr(scenarioShort), ...
    'FontName','Arial', ...
    'FontSize',8.2, ...
    'Box','off');

xlabel(axB,'reduction relative to frozen M2 baseline (%)','FontName','Arial');
title(axB,'B  Susceptible headcount versus edge-weighted burden', ...
    'HorizontalAlignment','left','FontName','Arial','FontWeight','bold');

grid(axB,'on');
set(axB,'GridAlpha',0.12);
axB.YGrid = 'off';

hHead = plot(axB,nan,nan,'o', ...
    'MarkerFaceColor',BLUE,'MarkerEdgeColor','w','MarkerSize',7);
hEdge = plot(axB,nan,nan,'s', ...
    'MarkerFaceColor',ORANGE,'MarkerEdgeColor','w','MarkerSize',6.7);

legend(axB,[hHead hEdge],{'maximum S_{head}','maximum S_{edge}'}, ...
    'Location','southeast', ...
    'Box','off', ...
    'FontName','Times New Roman', ...
    'FontSize',7.6);

text(axB,0.02,0.02,'greater separation indicates selective removal of edge-weighted burden', ...
    'Units','normalized','FontName','Arial','FontSize',6.8, ...
    'Color',[0.35 0.35 0.35]);

%% ------------------------------------------------------------------------
% Panel C. All-mechanism map in headcount-edge space
% -------------------------------------------------------------------------
hold(axC,'on');

x = med(:,3);
y = med(:,4);

lo = min([-5; x(:); y(:)]);
hi = max([65; x(:); y(:)]);
lo = floor(lo/5)*5;
hi = ceil(hi/5)*5;

plot(axC,[lo hi],[lo hi],'--','Color',GRAY,'LineWidth',0.9);

for i = 1:K
    plot(axC,x(i),y(i),'o', ...
        'MarkerFaceColor',MECH_COL(i,:), ...
        'MarkerEdgeColor','w', ...
        'MarkerSize',8.5);

    [dx,dy] = localLabelOffset(i);
    text(axC,x(i)+dx,y(i)+dy,scenarioShort{i}, ...
        'FontName','Arial','FontSize',7.0, ...
        'Color',MECH_COL(i,:), ...
        'FontWeight','bold');
end

% Emphasise targeting and immunity medians with diamond outlines.
plot(axC,x(targetIndex),y(targetIndex),'d', ...
    'MarkerFaceColor',MECH_COL(targetIndex,:), ...
    'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',0.8);

plot(axC,x(immuneIndex),y(immuneIndex),'d', ...
    'MarkerFaceColor',MECH_COL(immuneIndex,:), ...
    'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',0.8);

xlim(axC,[lo hi]);
ylim(axC,[lo hi]);
axis(axC,'square');

xlabel(axC,'reduction in maximum S_{head} (%)','FontName','Times New Roman');
ylabel(axC,'reduction in maximum S_{edge} (%)','FontName','Times New Roman');

title(axC,'C  Mechanism map', ...
    'HorizontalAlignment','left','FontName','Arial','FontWeight','bold');

grid(axC,'on');
set(axC,'GridAlpha',0.12,'FontName','Arial','FontSize',8.2,'Box','off');

text(axC,0.04,0.96, ...
    'above identity = greater reduction in edge burden than headcount', ...
    'Units','normalized','FontName','Arial','FontSize',6.8, ...
    'Color',[0.35 0.35 0.35], ...
    'VerticalAlignment','top');

%% ------------------------------------------------------------------------
% Save figure
% -------------------------------------------------------------------------
savefig(fig,fullfile(FIG_DIR,'Figure5_mechanism_counterfactuals.fig'));
localExport(fig,fullfile(FIG_DIR,'Figure5_mechanism_counterfactuals'));

%% Source data
outSource = table();
for i = 1:K
    T = rowsByScenario{i};
    tmp = T(:,requiredVars);
    outSource = [outSource;tmp]; %#ok<AGROW>
end
writetable(outSource,fullfile(SRC_DIR,'Figure5_source_data.csv'));

%% ------------------------------------------------------------------------
% Generate Main Table 3
% -------------------------------------------------------------------------
tableRows = table();

for i = 1:K
    row = table({scenarioShort{i}},{scenarioTable{i}}, ...
        med(i,1),q25(i,1),q75(i,1), ...
        med(i,2),q25(i,2),q75(i,2), ...
        med(i,3),q25(i,3),q75(i,3), ...
        med(i,4),q25(i,4),q75(i,4), ...
        'VariableNames',{ ...
        'scenario','model_perturbation', ...
        'peak_median','peak_q25','peak_q75', ...
        'cumulative_median','cumulative_q25','cumulative_q75', ...
        'Shead_median','Shead_q25','Shead_q75', ...
        'Sedge_median','Sedge_q25','Sedge_q75'});
    tableRows = [tableRows;row]; %#ok<AGROW>
end

writetable(tableRows,fullfile(TABLE_DIR,'main_table3_counterfactuals_final.csv'));

texFile = fullfile(TABLE_DIR,'main_table3_counterfactuals_final.tex');
fid = fopen(texFile,'w');

if fid < 0
    error('Could not open Table 3 TeX output.');
end

fprintf(fid,'\\begin{table}[p]\n');
fprintf(fid,'\\centering\n');
fprintf(fid,'\\caption{Conditional mechanism counterfactuals across seven representative fitted waves}\n');
fprintf(fid,'\\label{tab:counterfactuals}\n');
fprintf(fid,'\\small\n');
fprintf(fid,'\\begin{tabularx}{\\textwidth}{p{2.7cm} X r r r r}\n');
fprintf(fid,'\\toprule\n');
fprintf(fid,'Scenario & Model perturbation & Peak incidence & Cumulative incidence & max $\\Shead$ & max $\\Sedge$\\\\\n');
fprintf(fid,'\\midrule\n');

for i = 1:K
    fprintf(fid,'%s & %s & %s & %s & %s & %s\\\\\n', ...
        scenarioShort{i},scenarioTable{i}, ...
        localMedianIQR(med(i,1),q25(i,1),q75(i,1)), ...
        localMedianIQR(med(i,2),q25(i,2),q75(i,2)), ...
        localMedianIQR(med(i,3),q25(i,3),q75(i,3)), ...
        localMedianIQR(med(i,4),q25(i,4),q75(i,4)));
end

fprintf(fid,'\\bottomrule\n');
fprintf(fid,'\\end{tabularx}\n');
fprintf(fid,'\\par\\medskip\n');
fprintf(fid,['\\footnotesize Values are median percentage reductions [IQR] relative to each wave''s frozen M2 baseline. ' ...
    'The seven representative waves are W072, IT04, JP06, KR06, UK03, US04, and ZA04. ' ...
    'Highest-activity targeting halves activity in the top three of 12 equal-mass activity classes (25\\%% of the model population) without renormalising the remaining activity values. ' ...
    'The faster-isolation counterfactual keeps the fitted initial state fixed and changes only post-baseline $\\gamma$. ' ...
    'These are conditional structural perturbations, not estimates of historical policy effects.\n']);
fprintf(fid,'\\end{table}\n');

fclose(fid);

disp(tableRows);
fprintf('Figure 5 saved to %s\n',FIG_DIR);
fprintf('Main Table 3 saved to %s\n',texFile);

%% ------------------------------------------------------------------------
% Local functions
% -------------------------------------------------------------------------
function c = localCellstr(x)
    if iscell(x)
        c = x;
    elseif iscategorical(x)
        c = cellstr(x);
    elseif ischar(x)
        c = cellstr(x);
    else
        c = cellstr(x);
    end
end

function q = localQuantile(x,p)
    x = x(isfinite(x));
    x = sort(x(:));
    n = numel(x);

    q = zeros(size(p));

    for k = 1:numel(p)
        if n == 0
            q(k) = NaN;
        elseif n == 1
            q(k) = x(1);
        else
            pos = 1 + (n-1)*p(k);
            low = floor(pos);
            high = ceil(pos);

            if low == high
                q(k) = x(low);
            else
                q(k) = x(low) + (pos-low)*(x(high)-x(low));
            end
        end
    end
end

function txt = localMedianIQR(medv,q25v,q75v)
    txt = sprintf('%.1f [%.1f--%.1f]\\%%',medv,q25v,q75v);
end

function [dx,dy] = localLabelOffset(i)
    % Manual offsets keep labels legible in the compact mechanism map.
    switch i
        case 1
            dx = 1.6; dy = 2.3;   % gating
        case 2
            dx = 1.6; dy = -4.0;  % contact
        case 3
            dx = 1.6; dy = 2.5;   % isolation
        case 4
            dx = 1.8; dy = -3.3;  % immunity
        case 5
            dx = 1.8; dy = 2.0;   % targeting
        case 6
            dx = 1.8; dy = 2.2;   % combined
        otherwise
            dx = 1.5; dy = 1.5;
    end
end

function localExport(fig,stem)
    if exist('exportgraphics','file') == 2
        exportgraphics(fig,[stem '.png'],'Resolution',300);
        exportgraphics(fig,[stem '.jpg'],'Resolution',300);
        exportgraphics(fig,[stem '.pdf'],'ContentType','vector');

        try
            exportgraphics(fig,[stem '.svg'],'ContentType','vector');
        catch
        end
    else
        print(fig,[stem '.png'],'-dpng','-r300');
        print(fig,[stem '.jpg'],'-djpeg','-r300');
        print(fig,[stem '.pdf'],'-dpdf','-painters');
    end
end
