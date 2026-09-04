%% F32_Figure3_representative_fits_final.m
% MATLAB R2024b-compatible version.
% Reads final frozen fitted curves; does not refit models.
% Key fixes:
%   1) Preserve original CSV column names.
%   2) Validate required columns explicitly.
%   3) Resolve project root robustly.
%   4) Avoid assumptions about table text storage type.

clear; close all; clc;

SCRIPT_DIR = fileparts(mfilename('fullpath'));
ROOT = localProjectRoot(SCRIPT_DIR);

RESULTS = fullfile(ROOT,'outputs','results');
FIG_DIR = fullfile(ROOT,'outputs','figures');
SRC_DIR = fullfile(FIG_DIR,'source_data');

if ~exist(FIG_DIR,'dir'), mkdir(FIG_DIR); end
if ~exist(SRC_DIR,'dir'), mkdir(SRC_DIR); end

intlSummaryFile = fullfile(RESULTS,'international_fit_summary.csv');
intlCurveFile   = fullfile(RESULTS,'international_fit_curves.csv');
chinaSummaryFile= fullfile(RESULTS,'china_fit_summary_111_three_models.csv');
chinaCurveFile  = fullfile(RESULTS,'china_fit_curves_111_three_models.csv');

assert(exist(intlSummaryFile,'file')==2,'Missing file: %s',intlSummaryFile);
assert(exist(intlCurveFile,'file')==2,'Missing file: %s',intlCurveFile);
assert(exist(chinaSummaryFile,'file')==2,'Missing file: %s',chinaSummaryFile);
assert(exist(chinaCurveFile,'file')==2,'Missing file: %s',chinaCurveFile);

intlS = readtable(intlSummaryFile,'VariableNamingRule','preserve');
intlC = readtable(intlCurveFile,'VariableNamingRule','preserve');
chinaS = readtable(chinaSummaryFile,'VariableNamingRule','preserve');
chinaC = readtable(chinaCurveFile,'VariableNamingRule','preserve');

localRequireVars(intlS, ...
    {'wave_id','variant','winner', ...
     'delta_aicc_classic_minus_reservoir', ...
     'delta_aicc_reservoir_minus_network'}, ...
    'international fit summary');

localRequireVars(chinaS, ...
    {'wave_id','variant','winner', ...
     'delta_aicc_classic_minus_reservoir', ...
     'delta_aicc_reservoir_minus_network'}, ...
    'China fit summary');

curveVars = {'wave_id','day','observed','classic_pred', ...
             'reservoir_pred','network_pred'};

localRequireVars(intlC,curveVars,'international fit curves');
localRequireVars(chinaC,curveVars,'China fit curves');

reps = {'W072','IT04','JP06','KR06','UK03','US04','ZA04'};
settings = {'China (Shanghai)','Italy','Japan','South Korea', ...
            'United Kingdom','United States','South Africa'};

COL_M0 = [0.47 0.47 0.47];
COL_M1 = [0.00 0.45 0.70];
COL_M2 = [0.84 0.37 0.00];

fig = figure('Color','w','Units','inches', ...
    'Position',[0.4 0.5 13.6 7.25]);

sourceRows = table();
repRows = table();

for k=1:numel(reps)
    wid = reps{k};

    if wid(1)=='W' || wid(1)=='X'
        S = chinaS;
        C = chinaC;
    else
        S = intlS;
        C = intlC;
    end

    sids = localCellstr(S.wave_id);
    cids = localCellstr(C.wave_id);

    r = S(strcmp(sids,wid),:);
    g = C(strcmp(cids,wid),:);
    g = sortrows(g,'day');

    if height(r)~=1 || isempty(g)
        error('%s: missing or duplicated final result.',wid);
    end

    ax = subplot(2,4,k);
    hold(ax,'on');

    obs = max(double(g.observed),0.8);
    m0  = max(double(g.classic_pred),0.8);
    m1  = max(double(g.reservoir_pred),0.8);
    m2  = max(double(g.network_pred),0.8);

    scatter(ax,double(g.day),obs,12,'k','filled');
    plot(ax,double(g.day),m0,'Color',COL_M0,'LineWidth',1.0);
    plot(ax,double(g.day),m1,'Color',COL_M1,'LineWidth',1.25);
    plot(ax,double(g.day),m2,'Color',COL_M2,'LineWidth',1.35);

    set(ax,'YScale','log','FontName','Arial','FontSize',8,'Box','off');
    grid(ax,'on');
    set(ax,'GridAlpha',0.13);

    xlabel(ax,'day','FontName','Arial');
    ylabel(ax,'7-day mean reported cases','FontName','Arial');

    variant = localText(r.variant);

    title(ax,sprintf('%c  %s: %s', ...
        char('A'+k-1),settings{k},variant), ...
        'FontName','Arial','FontWeight','bold','FontSize',9.3, ...
        'HorizontalAlignment','left');

    h = localH(r);
    d01 = double(r.delta_aicc_classic_minus_reservoir(1));
    d12 = double(r.delta_aicc_reservoir_minus_network(1));
    winner = localModelCode(localText(r.winner));

    ann = sprintf( ...
        'winner=%s   h=%.2g   dAICc M0-M1=%.1f   dAICc M1-M2=%.1f', ...
        winner,h,d01,d12);

    text(ax,0.035,0.94,ann, ...
        'Units','normalized','VerticalAlignment','top', ...
        'FontName','Times New Roman','FontSize',7.0, ...
        'Color',[0.18 0.18 0.18]);

    tmp = table( ...
        repmat({wid},height(g),1), ...
        repmat({settings{k}},height(g),1), ...
        repmat({variant},height(g),1), ...
        double(g.day), ...
        double(g.observed), ...
        double(g.classic_pred), ...
        double(g.reservoir_pred), ...
        double(g.network_pred), ...
        'VariableNames', ...
        {'wave_id','setting','variant','day','observed', ...
         'M0_classic_pred','M1_reservoir_pred', ...
         'M2_activity_stratified_pred'});

    sourceRows = [sourceRows;tmp]; %#ok<AGROW>

    repTmp = table( ...
        {wid},{settings{k}},{variant},{localText(r.winner)},{winner}, ...
        h,d01,d12,height(g), ...
        'VariableNames', ...
        {'wave_id','setting','variant','winner','winner_code', ...
         'h','delta_AICc_M0_minus_M1', ...
         'delta_AICc_M1_minus_M2','n_days'});

    repRows = [repRows;repTmp]; %#ok<AGROW>
end

axL = subplot(2,4,8);
axis(axL,'off');
hold(axL,'on');

hh = gobjects(4,1);
hh(1) = scatter(axL,nan,nan,24,'k','filled');
hh(2) = plot(axL,nan,nan,'Color',COL_M0,'LineWidth',1.3);
hh(3) = plot(axL,nan,nan,'Color',COL_M1,'LineWidth',1.5);
hh(4) = plot(axL,nan,nan,'Color',COL_M2,'LineWidth',1.6);

legend(axL,hh, ...
    {'observed centred 7-day mean', ...
     'M0 classic SEIR', ...
     'M1 homogeneous RA-SEIR', ...
     'M2 activity-stratified RA-SEIR'}, ...
    'Box','off','FontName','Arial','FontSize',9, ...
    'Location','northwest');

text(axL,.08,.35, ...
    sprintf(['Positive dAICc values favour the model to the right.\n' ...
             'Fitted scale Q is an observation/accessibility scale,\n' ...
             'not census population.']), ...
    'Units','normalized','FontName','Times New Roman','FontSize',8);

annotation(fig,'textbox',[0.07 0.965 0.86 0.03], ...
    'String', ...
    'Representative fits show when dynamic recruitment and contact heterogeneity alter epidemic-wave shape', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontWeight','bold','FontSize',14);

savefig(fig,fullfile(FIG_DIR,'Figure3_representative_fits.fig'));
localExport(fig,fullfile(FIG_DIR,'Figure3_representative_fits'));

writetable(sourceRows,fullfile(SRC_DIR,'Figure3_source_data.csv'));
writetable(repRows,fullfile(SRC_DIR,'Figure3_representatives.csv'));

disp(repRows);

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

function txt = localText(x)
    c = localCellstr(x);
    txt = c{1};
end

function h = localH(r)
    vars = r.Properties.VariableNames;

    if any(strcmp(vars,'network_h_grid'))
        val = r.network_h_grid;
        if ~isempty(val) && isfinite(double(val(1)))
            h = double(val(1));
            return
        end
    end

    if any(strcmp(vars,'network_h_cv2'))
        h = double(r.network_h_cv2(1));
        return
    end

    error('Neither network_h_grid nor network_h_cv2 is available.');
end

function code = localModelCode(w)
    if strcmp(w,'classic')
        code = 'M0';
    elseif strcmp(w,'reservoir')
        code = 'M1';
    elseif strcmp(w,'network')
        code = 'M2';
    else
        code = w;
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
