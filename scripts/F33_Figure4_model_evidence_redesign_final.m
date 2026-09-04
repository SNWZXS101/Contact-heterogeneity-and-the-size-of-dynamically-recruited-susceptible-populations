%% F33_Figure4_model_evidence_redesign_final.m
% MATLAB R2024b-compatible version.
%
% Redesigned Figure 4:
% A 100% horizontal stacked composition
% B evidence-threshold dumbbell/dot plot
% C jitter + median/IQR of max(Sedge)/max(Shead) across h
% D ranked held-out RMSE difference
%
% Key fixes:
%   1) Preserve original CSV column names.
%   2) Rename table variable "hold" -> "holdT" so it does not shadow hold().
%   3) Use graphics-object arrays instead of numeric handle arrays.
%   4) Validate required columns explicitly.
%   5) Resolve project root robustly.
%
% Reads final frozen outputs only.

clear; close all; clc;
rng(1);

SCRIPT_DIR = fileparts(mfilename('fullpath'));
ROOT = localProjectRoot(SCRIPT_DIR);

RESULTS = fullfile(ROOT,'outputs','results');
FIG_DIR = fullfile(ROOT,'outputs','figures');
SRC_DIR = fullfile(FIG_DIR,'source_data');

if ~exist(FIG_DIR,'dir'), mkdir(FIG_DIR); end
if ~exist(SRC_DIR,'dir'), mkdir(SRC_DIR); end

intlFile = fullfile(RESULTS,'international_fit_summary.csv');
chinaFile = fullfile(RESULTS,'china_fit_summary_111_three_models.csv');
holdoutFile = fullfile(RESULTS,'international_holdout_metrics.csv');

assert(exist(intlFile,'file')==2,'Missing file: %s',intlFile);
assert(exist(chinaFile,'file')==2,'Missing file: %s',chinaFile);
assert(exist(holdoutFile,'file')==2,'Missing file: %s',holdoutFile);

intl = readtable(intlFile,'VariableNamingRule','preserve');
china = readtable(chinaFile,'VariableNamingRule','preserve');
holdT = readtable(holdoutFile,'VariableNamingRule','preserve');

summaryVars = {'wave_id','winner', ...
               'delta_aicc_classic_minus_reservoir', ...
               'delta_aicc_reservoir_minus_network'};

localRequireVars(intl,summaryVars,'international fit summary');
localRequireVars(china,summaryVars,'China fit summary');

localRequireVars(holdT, ...
    {'wave_id','country','classic_test_log_rmse', ...
     'reservoir_family_test_log_rmse'}, ...
    'international holdout metrics');

assert(numel(unique(localCellstr(intl.wave_id)))==42, ...
    'Expected 42 international waves.');
assert(numel(unique(localCellstr(china.wave_id)))==111, ...
    'Expected 111 Chinese waves.');
assert(numel(unique(localCellstr(holdT.wave_id)))==42, ...
    'Expected 42 holdout waves.');

COL_M0 = [0.47 0.47 0.47];
COL_M1 = [0.00 0.45 0.70];
COL_M2 = [0.84 0.37 0.00];
COL_CHINA = [0.30 0.55 0.78];
COL_INTL = [0.90 0.45 0.15];

countryNames = { ...
    'Italy','Japan','South Africa','South Korea', ...
    'United Kingdom','United States'};

countryColors = [ ...
    0.30 0.47 0.67
    0.95 0.58 0.20
    0.35 0.63 0.35
    0.88 0.33 0.34
    0.69 0.48 0.66
    0.61 0.46 0.37
];

fig = figure('Color','w','Units','inches', ...
    'Position',[0.35 0.35 13.5 9.0]);

%% Panel A
axA = subplot(2,2,1);
hold(axA,'on');

cw = localWinnerCounts(china);
iw = localWinnerCounts(intl);

dataCounts = [cw; iw];
N = sum(dataCounts,2);
pct = 100*dataCounts./N;

b = barh(axA,1:2,pct,'stacked','BarWidth',0.58);

set(b(1),'FaceColor',COL_M0);
set(b(2),'FaceColor',COL_M1);
set(b(3),'FaceColor',COL_M2);

for i=1:2
    cum = 0;

    for j=1:3
        if pct(i,j)>=3
            col = [1 1 1];

            if j==1
                col = [0.05 0.05 0.05];
            end

            text(axA,cum+pct(i,j)/2,i, ...
                sprintf('%d\n(%.0f%%)',dataCounts(i,j),pct(i,j)), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontName','Arial','FontWeight','bold','FontSize',8, ...
                'Color',col);
        end

        cum = cum+pct(i,j);
    end
end

xlim(axA,[0 100]);

set(axA,'XTick',0:20:100,'YTick',[1 2], ...
    'YTickLabel',{'International national','China local/provincial'}, ...
    'FontName','Arial','FontSize',8,'Box','off');

xlabel(axA,'share of waves (%)','FontName','Arial');

title(axA,'A  Model winners by observation scale', ...
    'HorizontalAlignment','left','FontName','Arial', ...
    'FontWeight','bold');

grid(axA,'on');
set(axA,'GridAlpha',0.12);

legend(axA,b, ...
    {'M0 classic SEIR','M1 homogeneous RA-SEIR', ...
     'M2 activity-stratified RA-SEIR'}, ...
    'Location','southoutside','Orientation','horizontal', ...
    'Box','off','FontName','Arial','FontSize',7.4);

%% Panel B
axB = subplot(2,2,2);
hold(axB,'on');

labels = { ...
    'Recruitment: China', ...
    'Recruitment: international', ...
    'Heterogeneity: China', ...
    'Heterogeneity: international'};

Tlist = {china,intl,china,intl};

cols = { ...
    'delta_aicc_classic_minus_reservoir', ...
    'delta_aicc_classic_minus_reservoir', ...
    'delta_aicc_reservoir_minus_network', ...
    'delta_aicc_reservoir_minus_network'};

p2 = zeros(4,1);
p10 = zeros(4,1);
n2 = zeros(4,1);
n10 = zeros(4,1);
nrow = zeros(4,1);

for i=1:4
    x = double(Tlist{i}.(cols{i}));
    x = x(isfinite(x));

    nrow(i) = numel(x);

    if nrow(i)==0
        error('No finite values available for Panel B comparison %d.',i);
    end

    n2(i) = sum(x>2);
    n10(i) = sum(x>10);

    p2(i) = 100*n2(i)/nrow(i);
    p10(i) = 100*n10(i)/nrow(i);
end

y = 4:-1:1;

for i=1:4
    plot(axB,[p10(i) p2(i)],[y(i) y(i)],'-', ...
        'Color',[0.72 0.72 0.72],'LineWidth',2.2);

    plot(axB,p2(i),y(i),'o','MarkerSize',8, ...
        'MarkerFaceColor',[0.12 0.42 0.68], ...
        'MarkerEdgeColor','w');

    plot(axB,p10(i),y(i),'s','MarkerSize',7, ...
        'MarkerFaceColor',[0.90 0.40 0.10], ...
        'MarkerEdgeColor','w');

    text(axB,p2(i)+2,y(i)+0.10, ...
        sprintf('%d/%d',n2(i),nrow(i)), ...
        'FontName','Arial','FontSize',7.2, ...
        'Color',[0.12 0.42 0.68]);

    if p10(i)>3
        text(axB,p10(i)-2,y(i)-0.15, ...
            sprintf('%d/%d',n10(i),nrow(i)), ...
            'HorizontalAlignment','right', ...
            'FontName','Arial','FontSize',7.2, ...
            'Color',[0.78 0.30 0.05]);
    end
end

xlim(axB,[0 105]);
ylim(axB,[0.4 4.6]);

set(axB,'XTick',0:20:100,'YTick',1:4, ...
    'YTickLabel',fliplr(labels), ...
    'FontName','Arial','FontSize',8,'Box','off');

xlabel(axB, ...
    'waves exceeding information-criterion threshold (%)', ...
    'FontName','Arial');

title(axB, ...
    'B  Strength of evidence by model step and observation scale', ...
    'HorizontalAlignment','left','FontName','Arial', ...
    'FontWeight','bold');

grid(axB,'on');
set(axB,'GridAlpha',0.12);

h1 = plot(axB,nan,nan,'o', ...
    'MarkerFaceColor',[0.12 0.42 0.68], ...
    'MarkerEdgeColor','w','MarkerSize',8);

h2 = plot(axB,nan,nan,'s', ...
    'MarkerFaceColor',[0.90 0.40 0.10], ...
    'MarkerEdgeColor','w','MarkerSize',7);

legend(axB,[h1 h2], ...
    {'dAICc > 2','dAICc > 10'}, ...
    'Location','southeast','Box','off', ...
    'FontName','Times New Roman','FontSize',8);

%% Panel C
axC = subplot(2,2,3);
hold(axC,'on');

[headC,edgeC,hC] = localHeadEdgeH(china);
[headI,edgeI,hI] = localHeadEdgeH(intl);

ratioC = edgeC./headC;
ratioI = edgeI./headI;

hgrid = [0 0.25 0.5 1 2 4];
xpos = 1:numel(hgrid);

for j=1:numel(hgrid)
    rc = ratioC(abs(hC-hgrid(j))<1e-8 & ...
        isfinite(ratioC) & ratioC>0);

    ri = ratioI(abs(hI-hgrid(j))<1e-8 & ...
        isfinite(ratioI) & ratioI>0);

    jc = (rand(size(rc))-0.5)*0.22;
    ji = (rand(size(ri))-0.5)*0.22;

    scatter(axC,xpos(j)-0.07+jc,rc,22,COL_CHINA,'filled');
    scatter(axC,xpos(j)+0.07+ji,ri,28,COL_INTL,'filled');

    rr = [rc;ri];

    if ~isempty(rr)
        q = localQuantile(rr,[0.25 0.50 0.75]);

        plot(axC,[xpos(j) xpos(j)],[q(1) q(3)], ...
            'k-','LineWidth',2.2);

        plot(axC,xpos(j),q(2),'kd', ...
            'MarkerFaceColor','k','MarkerSize',6);
    end
end

plot(axC,[0.5 numel(hgrid)+0.5],[1 1],'--', ...
    'Color',[0.45 0.45 0.45],'LineWidth',1);

set(axC,'YScale','log','XTick',xpos, ...
    'XTickLabel',{'0','0.25','0.5','1','2','4'}, ...
    'FontName','Arial','FontSize',8,'Box','off');

xlabel(axC,'heterogeneity index, h = CV^2(Z)', ...
    'FontName','Times New Roman');

ylabel(axC,'maximum S_{edge} / maximum S_{head}', ...
    'FontName','Times New Roman');

title(axC, ...
    'C  Headcount-edge divergence increases with heterogeneity', ...
    'HorizontalAlignment','left','FontName','Arial', ...
    'FontWeight','bold');

grid(axC,'on');
set(axC,'GridAlpha',0.12);

hc = scatter(axC,nan,nan,24,COL_CHINA,'filled');
hi = scatter(axC,nan,nan,28,COL_INTL,'filled');
hm = plot(axC,nan,nan,'kd','MarkerFaceColor','k','MarkerSize',6);

legend(axC,[hc hi hm], ...
    {'China local/provincial','International national', ...
     'median (IQR line)'}, ...
    'Location','northwest','Box','off', ...
    'FontName','Arial','FontSize',7.3);

%% Panel D
axD = subplot(2,2,4);
hold(axD,'on');

m0 = double(holdT.classic_test_log_rmse);
fam = double(holdT.reservoir_family_test_log_rmse);
delta = fam-m0;

ok = isfinite(delta);

H = holdT(ok,:);
m0 = m0(ok);
fam = fam(ok);
delta = delta(ok);

[deltaS,ord] = sort(delta,'ascend');

H = H(ord,:);
m0 = m0(ord);
fam = fam(ord);

rankIndex = (1:numel(deltaS))';

plot(axD,[0 0],[0 numel(deltaS)+1],'--', ...
    'Color',[0.35 0.35 0.35],'LineWidth',1);

holdCountries = localCellstr(H.country);
holdIDs = localCellstr(H.wave_id);

for i=1:numel(deltaS)
    col = localCountryColor( ...
        holdCountries{i},countryNames,countryColors);

    plot(axD,[0 deltaS(i)],[rankIndex(i) rankIndex(i)],'-', ...
        'Color',[0.80 0.80 0.80],'LineWidth',0.8);

    plot(axD,deltaS(i),rankIndex(i),'o','MarkerSize',5.7, ...
        'MarkerFaceColor',col,'MarkerEdgeColor','w');
end

set(axD,'YDir','reverse','YTick',[], ...
    'FontName','Arial','FontSize',8,'Box','off');

xlabel(axD, ...
    'training-selected RA-SEIR family - M0 test RMSE_{log}', ...
    'FontName','Times New Roman');

ylabel(axD, ...
    '42 international waves, ranked by prediction gain', ...
    'FontName','Arial');

title(axD, ...
    'D  Tail-prediction gains are heterogeneous across waves', ...
    'HorizontalAlignment','left','FontName','Arial', ...
    'FontWeight','bold');

grid(axD,'on');
set(axD,'GridAlpha',0.12);

beats = sum(deltaS<0);

text(axD,0.03,0.06, ...
    sprintf('RA-SEIR family lower test error in %d/%d waves', ...
    beats,numel(deltaS)), ...
    'Units','normalized','FontName','Arial', ...
    'FontWeight','bold','FontSize',8.2);

idxLab = unique([ ...
    1:min(3,numel(deltaS)), ...
    max(1,numel(deltaS)-2):numel(deltaS)]);

for kk=1:numel(idxLab)
    ii = idxLab(kk);

    text(axD,deltaS(ii),rankIndex(ii),['  ' holdIDs{ii}], ...
        'FontName','Arial','FontSize',6.8, ...
        'VerticalAlignment','middle');
end

hh = gobjects(numel(countryNames),1);

for i=1:numel(countryNames)
    hh(i) = plot(axD,nan,nan,'o', ...
        'MarkerFaceColor',countryColors(i,:), ...
        'MarkerEdgeColor','w','MarkerSize',5.7);
end

legend(axD,hh,countryNames, ...
    'Location','southeast','NumColumns',2, ...
    'Box','off','FontName','Arial','FontSize',6.6);

annotation(fig,'textbox',[0.05 0.963 0.90 0.03], ...
    'String', ...
    'Empirical support for dynamic recruitment and contact heterogeneity is strong but heterogeneous across waves', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontWeight','bold','FontSize',14);

savefig(fig,fullfile(FIG_DIR,'Figure4_model_evidence.fig'));
localExport(fig,fullfile(FIG_DIR,'Figure4_model_evidence'));

panelA = table( ...
    {'China local/provincial';'International national'}, ...
    sum(dataCounts,2), ...
    dataCounts(:,1), ...
    dataCounts(:,2), ...
    dataCounts(:,3), ...
    'VariableNames',{'analysis_set','n','M0','M1','M2'});

writetable(panelA, ...
    fullfile(SRC_DIR,'Figure4_panelA_counts.csv'));

panelB = table( ...
    labels',nrow,n2,p2,n10,p10, ...
    'VariableNames', ...
    {'comparison','n','n_gt2','pct_gt2','n_gt10','pct_gt10'});

writetable(panelB, ...
    fullfile(SRC_DIR,'Figure4_panelB_thresholds.csv'));

panelC = table( ...
    [hC;hI], ...
    [ratioC;ratioI], ...
    [repmat({'China local/provincial'},numel(hC),1); ...
     repmat({'International national'},numel(hI),1)], ...
    'VariableNames', ...
    {'h','max_Sedge_over_max_Shead','analysis_scale'});

writetable(panelC, ...
    fullfile(SRC_DIR,'Figure4_panelC_head_edge_ratio.csv'));

panelD = table( ...
    holdIDs,m0,fam,deltaS, ...
    'VariableNames', ...
    {'wave_id','M0_RMSElog','RA_family_RMSElog','family_minus_M0'});

writetable(panelD, ...
    fullfile(SRC_DIR,'Figure4_panelD_holdout_difference.csv'));

fprintf('Figure 4 saved to %s\n',FIG_DIR);

function ROOT = localProjectRoot(scriptDir)
    if exist(fullfile(scriptDir,'outputs'),'dir')==7
        ROOT = scriptDir;
        return
    end

    parentDir = fileparts(scriptDir);

    if exist(fullfile(parentDir,'outputs'),'dir')==7
        ROOT = parentDir;
        return
    end

    error(['Cannot identify project root. Expected outputs/ either beside ' ...
           'this script or one directory above it.']);
end

function localRequireVars(T,names,label)
    vars = T.Properties.VariableNames;
    missing = names(~ismember(names,vars));

    if ~isempty(missing)
        error('%s is missing required column(s): %s', ...
            label,strjoin(missing,', '));
    end
end

function counts = localWinnerCounts(T)
    w = localCellstr(T.winner);

    counts = [ ...
        sum(strcmp(w,'classic')) ...
        sum(strcmp(w,'reservoir')) ...
        sum(strcmp(w,'network'))];
end

function [head,edge,h] = localHeadEdgeH(T)
    vars = T.Properties.VariableNames;

    if all(ismember( ...
            {'network_peak_head_fraction_Q', ...
             'network_peak_edge_fraction_Q'},vars))

        head = double(T.network_peak_head_fraction_Q);
        edge = double(T.network_peak_edge_fraction_Q);

    elseif all(ismember( ...
            {'network_max_S_head','network_max_S_edge','network_Q'},vars))

        head = double(T.network_max_S_head)./double(T.network_Q);
        edge = double(T.network_max_S_edge)./double(T.network_Q);

    else
        error(['Required head/edge columns are absent. Need either ' ...
               'network_peak_head_fraction_Q + network_peak_edge_fraction_Q ' ...
               'or network_max_S_head + network_max_S_edge + network_Q.']);
    end

    if any(strcmp(vars,'network_h_grid'))
        h = double(T.network_h_grid);
    elseif any(strcmp(vars,'network_h_cv2'))
        h = double(T.network_h_cv2);
    else
        error('Neither network_h_grid nor network_h_cv2 is available.');
    end
end

function q = localQuantile(x,p)
    x = sort(x(:));
    n = numel(x);
    q = zeros(size(p));

    for k=1:numel(p)
        if n==1
            q(k) = x(1);
        else
            pos = 1+(n-1)*p(k);
            lo = floor(pos);
            hi = ceil(pos);

            if lo==hi
                q(k) = x(lo);
            else
                q(k) = x(lo)+(pos-lo)*(x(hi)-x(lo));
            end
        end
    end
end

function c = localCountryColor(country,names,colors)
    idx = find(strcmp(names,country),1);

    if isempty(idx)
        c = [0.35 0.35 0.35];
    else
        c = colors(idx,:);
    end
end

function c = localCellstr(x)
    if iscell(x)
        c = x;
    elseif iscategorical(x)
        c = cellstr(x);
    elseif ischar(x)
        c = cellstr(x);
    elseif isstring(x)
        c = cellstr(x);
    else
        c = cellstr(string(x));
    end
end

function localExport(fig,stem)
    if exist('exportgraphics','file')==2
        exportgraphics(fig,[stem '.png'],'Resolution',300);
        exportgraphics(fig,[stem '.pdf'],'ContentType','vector');

        try
            exportgraphics(fig,[stem '.svg'],'ContentType','vector');
        catch
        end
    else
        print(fig,[stem '.png'],'-dpng','-r300');
        print(fig,[stem '.pdf'],'-dpdf','-painters');
    end
end
