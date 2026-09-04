%% F31_Figure2_multicountry_trajectories_final.m
% MATLAB R2024b-compatible version.
% Key fixes:
%   1) Preserve original CSV column names.
%   2) Access reserved column name "end" with dynamic table indexing.
%   3) Use graphics-object arrays instead of numeric handle arrays.
%   4) Resolve project root whether this file is in project root or scripts/.
%
% Editable output:
%   outputs/figures/Figure2_multicountry_trajectories.fig

clear; close all; clc;

SCRIPT_DIR = fileparts(mfilename('fullpath'));
ROOT = localProjectRoot(SCRIPT_DIR);

SOURCE_DIR = fullfile(ROOT,'data','source');
INTL_DIR   = fullfile(ROOT,'data','international');
RESULTS    = fullfile(ROOT,'outputs','results');
FIG_DIR    = fullfile(ROOT,'outputs','figures');
SRC_DIR    = fullfile(FIG_DIR,'source_data');

if ~exist(FIG_DIR,'dir'), mkdir(FIG_DIR); end
if ~exist(SRC_DIR,'dir'), mkdir(SRC_DIR); end

owidFile    = fullfile(SOURCE_DIR,'owid_selected_countries.csv');
catalogFile = fullfile(INTL_DIR,'international_wave_catalog.csv');
chinaFile   = fullfile(RESULTS,'china_fit_summary_111_three_models.csv');

assert(exist(owidFile,'file')==2,    'Missing file: %s',owidFile);
assert(exist(catalogFile,'file')==2, 'Missing file: %s',catalogFile);
assert(exist(chinaFile,'file')==2,   'Missing file: %s',chinaFile);

owid = readtable(owidFile,'VariableNamingRule','preserve');
catalog = readtable(catalogFile,'VariableNamingRule','preserve');
china = readtable(chinaFile,'VariableNamingRule','preserve');

localRequireVars(owid, ...
    {'country','date','new_cases_smoothed_per_million', ...
     'stringency_index','people_fully_vaccinated_per_hundred'}, ...
    'OWID source');

localRequireVars(catalog, ...
    {'wave_id','country','start','end','peak_date'}, ...
    'international wave catalog');

localRequireVars(china, ...
    {'wave_id','start','end'}, ...
    'China fit summary');

owid.date = localToDatetime(owid.date);
catalog.start = localToDatetime(catalog.start);
catalog.('end') = localToDatetime(catalog.('end'));
catalog.peak_date = localToDatetime(catalog.peak_date);
china.start = localToDatetime(china.start);
china.('end') = localToDatetime(china.('end'));

assert(numel(unique(localCellstr(catalog.wave_id)))==42, ...
    'Expected 42 international waves.');
assert(numel(unique(localCellstr(china.wave_id)))==111, ...
    'Expected 111 Chinese waves.');

countryOrder = {'China','United States','United Kingdom','Japan', ...
                'South Korea','Italy','South Africa'};

catalogEnd = catalog.('end');
chinaEnd = china.('end');

finalAnalysisDay = max([max(catalogEnd), max(chinaEnd)]);
availableEnd = max(owid.date);

if availableEnd < finalAnalysisDay
    error('OWID source snapshot ends before the final analysed wave.');
end

startDate = datetime(2020,1,1);
endDate = min(finalAnalysisDay + days(10),availableEnd);

countryCol = localCellstr(owid.country);
keepCountry = false(height(owid),1);
for i=1:numel(countryOrder)
    keepCountry = keepCountry | strcmp(countryCol,countryOrder{i});
end

keep = keepCountry & owid.date>=startDate & owid.date<=endDate;
P = owid(keep,:);

casesAll = double(P.new_cases_smoothed_per_million);
casesAll(~isfinite(casesAll)) = 0;
casesAll = max(casesAll,0);

displayFloor = 0.03;
globalMax = max(casesAll);
if isempty(globalMax) || ~isfinite(globalMax) || globalMax<=0
    globalMax = 1;
end
yTop = 10^ceil(log10(globalMax*1.08));

CASE_FILL = [157 183 211]/255;
CASE_LINE = [47 93 138]/255;
STRINGENCY = [213 94 0]/255;
VACC = [0 158 115]/255;
WINDOW = [184 199 217]/255;
PEAK = [0.45 0.45 0.45];

fig = figure('Color','w','Units','inches', ...
    'Position',[0.4 0.4 13.2 11.8]);

ax = gobjects(7,1);
legendHandles = gobjects(5,1);

pCountry = localCellstr(P.country);
cCountry = localCellstr(catalog.country);

for k=1:numel(countryOrder)
    country = countryOrder{k};
    D = P(strcmp(pCountry,country),:);

    if isempty(D)
        error('No plotting data for %s.',country);
    end

    ax(k) = subplot(7,1,k);
    hold(ax(k),'on');

    x = datenum(D.date);
    y = double(D.new_cases_smoothed_per_million);
    y(~isfinite(y)) = 0;
    y = max(y,displayFloor);

    if strcmp(country,'China')
        rugTop = displayFloor*1.9;
        for ii=1:height(china)
            x1 = datenum(china.start(ii));
            x2 = datenum(chinaEnd(ii));
            patch(ax(k),[x1 x2 x2 x1], ...
                [displayFloor displayFloor rugTop rugTop], ...
                WINDOW,'FaceAlpha',0.42,'EdgeColor','none');
        end
    else
        G = catalog(strcmp(cCountry,country),:);
        Gend = G.('end');

        for ii=1:height(G)
            x1 = datenum(G.start(ii));
            x2 = datenum(Gend(ii));

            patch(ax(k),[x1 x2 x2 x1], ...
                [displayFloor displayFloor yTop yTop], ...
                WINDOW,'FaceAlpha',0.10,'EdgeColor','none');

            xp = datenum(G.peak_date(ii));
            plot(ax(k),[xp xp],[displayFloor yTop],':', ...
                'Color',PEAK,'LineWidth',0.55);
        end
    end

    hFill = fill(ax(k),[x; flipud(x)], ...
        [repmat(displayFloor,size(x)); flipud(y)], ...
        CASE_FILL,'FaceAlpha',0.42,'EdgeColor','none');

    plot(ax(k),x,y,'Color',CASE_LINE,'LineWidth',0.95);

    set(ax(k),'YScale','log','YLim',[displayFloor yTop], ...
        'FontName','Arial','FontSize',8,'Box','off');
    grid(ax(k),'on');
    set(ax(k),'GridAlpha',0.12);

    ylabel(ax(k),{'7-day mean cases','per million'}, ...
        'FontName','Arial');

    yyaxis(ax(k),'right');

    s = double(D.stringency_index);
    v = double(D.people_fully_vaccinated_per_hundred);

    hString = plot(ax(k),x,s,'Color',STRINGENCY,'LineWidth',0.9);
    hVacc = plot(ax(k),x,v,'Color',VACC,'LineWidth',1.0);

    ylim(ax(k),[0 105]);
    set(ax(k),'YTick',[0 50 100]);

    yyaxis(ax(k),'left');

    if strcmp(country,'China')
        panelSubtitle = 'national context; 111 local/provincial waves';
    else
        n = sum(strcmp(cCountry,country));
        panelSubtitle = sprintf('%d analysed national waves',n);
    end

    text(ax(k),0.006,0.88, ...
        sprintf('%c  %s',char('A'+k-1),country), ...
        'Units','normalized','FontName','Arial', ...
        'FontWeight','bold','FontSize',9.1, ...
        'VerticalAlignment','top');

    text(ax(k),0.006,0.68,panelSubtitle, ...
        'Units','normalized','FontName','Arial','FontSize',7.0, ...
        'Color',[0.32 0.32 0.32],'VerticalAlignment','top');

    xlim(ax(k),[datenum(startDate) datenum(endDate)]);

    if k<numel(countryOrder)
        set(ax(k),'XTickLabel',[]);
    end

    if k==1
        legendHandles(1) = hFill;
        legendHandles(2) = hString;
        legendHandles(3) = hVacc;
        legendHandles(4) = patch(ax(k),nan,nan,WINDOW, ...
            'FaceAlpha',0.18,'EdgeColor','none');
        legendHandles(5) = plot(ax(k),nan,nan,':', ...
            'Color',PEAK,'LineWidth',1.0);
    end
end

datetick(ax(end),'x','mmm yyyy','keeplimits');

annotation(fig,'textbox',[0.08 0.965 0.84 0.03], ...
    'String', ...
    'Epidemic trajectories, vaccination, policy intensity, and analysed wave windows', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontWeight','bold','FontSize',14);

legend(ax(1),legendHandles, ...
    {'7-day mean reported cases per million', ...
     'Oxford stringency index', ...
     'fully vaccinated (%)', ...
     'analysed national-wave window', ...
     'nominated national-wave peak'}, ...
    'Orientation','horizontal','Box','off','FontName','Arial', ...
    'FontSize',7.4,'Location','northoutside');

annotation(fig,'textbox',[0.18 0.006 0.64 0.025], ...
    'String',['China is shown as national context for 111 local/provincial waves; ' ...
              'full-height shading denotes the 42 analysed national waves.'], ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontSize',7.3, ...
    'Color',[0.28 0.28 0.28]);

savefig(fig,fullfile(FIG_DIR,'Figure2_multicountry_trajectories.fig'));
localExport(fig,fullfile(FIG_DIR,'Figure2_multicountry_trajectories'));

writetable(P(:,{'country','date','new_cases_smoothed_per_million', ...
    'stringency_index','people_fully_vaccinated_per_hundred'}), ...
    fullfile(SRC_DIR,'Figure2_source_data.csv'));

fprintf('Figure 2 saved to %s\n',FIG_DIR);

function ROOT = localProjectRoot(scriptDir)
    if exist(fullfile(scriptDir,'data'),'dir')==7 && ...
            exist(fullfile(scriptDir,'outputs'),'dir')==7
        ROOT = scriptDir;
        return
    end

    parentDir = fileparts(scriptDir);
    if exist(fullfile(parentDir,'data'),'dir')==7 && ...
            exist(fullfile(parentDir,'outputs'),'dir')==7
        ROOT = parentDir;
        return
    end

    error(['Cannot identify project root. Expected data/ and outputs/ ' ...
           'either beside this script or one directory above it.']);
end

function localRequireVars(T,names,label)
    vars = T.Properties.VariableNames;
    missing = names(~ismember(names,vars));
    if ~isempty(missing)
        error('%s is missing required column(s): %s', ...
            label,strjoin(missing,', '));
    end
end

function dt = localToDatetime(x)
    if isdatetime(x)
        dt = x;
    elseif isnumeric(x)
        dt = datetime(x,'ConvertFrom','datenum');
    else
        dt = datetime(localCellstr(x));
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
    pngFile = [stem '.png'];
    pdfFile = [stem '.pdf'];
    svgFile = [stem '.svg'];

    if exist('exportgraphics','file')==2
        exportgraphics(fig,pngFile,'Resolution',300);
        exportgraphics(fig,pdfFile,'ContentType','vector');
        try
            exportgraphics(fig,svgFile,'ContentType','vector');
        catch
        end
    else
        print(fig,pngFile,'-dpng','-r300');
        print(fig,pdfFile,'-dpdf','-painters');
    end
end
