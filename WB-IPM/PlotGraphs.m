function PlotGraphs(TYPE,PS,x_true,n,maxit1,maxit2)
%
% PlotGraphs(TYPE,PS,x_true,n,maxit)
%
% This function is to plot the results of hybrid methods in MixHyBR
% project.
%
% Input:
%       TYPE - there are three type of plots ['recon','diff','relerr']
%              'recon' - plot reconstructions
%               'diff' - plot inverse reconstructions if reconstructions
%                        are undistinguishable
%             'relerr' - plot relative errors
%         PS - structure including hybrid,reg,x,output
%     x_true - true image
%          n - original size of true image if it is square
%      maxit1 - maximum number of iteration in hybrid methods
%      maxit2 - maximum number of iteration in hybrid recycle methods
%
% T.Cho Nov/2019
%
Length_St = size(PS,1);

if Length_St == 5
    C_St = 2;
    R_St = 2;
elseif Length_St == 4
    C_St = 1;
    R_St = 2;
elseif Length_St == 7
    C_St = 3;
    R_St = 2;
elseif Length_St == 6
    C_St = 2;
    R_St = 2;
else
    C_St = 2;
    R_St = ceil(Length_St/C_St);
end

switch TYPE
    
    case 'recon'
        figure,
        axisMin = min(x_true(:));
        axisMax = max(x_true(:));
        for L = 1:Length_St+1
            % Plot reconstructions
            if L == (Length_St+1) && (strcmp(PS{L-1,2},'optimal')==1)
                subplot(R_St,C_St+1,L),
                imagesc(reshape(x_true,n,n)),
                title('true field'), cbh=colorbar; caxis([axisMin, axisMax]), set(gca,'xtick',[],'ytick',[]);
                a =get(cbh);
                a =  a.Position; set(cbh,'Position',[a(1)+.08, a(2)+0.02, 0.05, 0.31]);
            elseif L < Length_St+1
                subplot(R_St,C_St+1,L), %imshow(reshape(PS{L,3},n,n),[]),
                imagesc(reshape(PS{L,3},n,n)),
                title([num2str(PS{L,1}),'-',num2str(PS{L,2}), ', iter=',num2str(PS{L,4}.iterations),' (',num2str(PS{L,4}.Enrm(PS{L,4}.iterations)),')'])
                colorbar, caxis([axisMin, axisMax]), colorbar off, set(gca,'xtick',[],'ytick',[]);
            end
            axis square
        end
        
        cbh = colorbar; a =get(cbh);
        a =  a.Position; set(cbh,'Position',[a(1)+0.08, a(2)-0.06, 0.04, 0.31],...
            'YTick',0:.2:1);
        cbh.FontSize = 10;
        
    case 'diff'
        
        f=figure;
        %scale = [0 1/norm(x_true)];
        for L = 1:Length_St
            if strcmp(PS{L,1},'mixHyBR') &&(strcmp(PS{L,2},'optimal')==0)
                x_recon = PS{L,4}.xstop;
            else
                x_recon = PS{L,3};
            end
            % Plot inverse reconstructions (if recons are undistinguishable)
            subplot(R_St,C_St+1,L), %imagesc(10*abs(reshape(x_recon,n,n)-x_true)/norm(x_true)),
            img = abs(reshape(x_recon,n,n)-x_true);
            imshow(img,[]),
            axis off, axis image,colormap('gray'); % scale = [0.4 1], caxis(scale)
            colormap(f,flipud(colormap(f))); caxis([0,0.6])
            title([num2str(PS{L,1}),'-',num2str(PS{L,2}),' (',num2str(PS{L,4}.Enrm(PS{L,4}.iterations)),')'],'fontsize',20)
            axis square
        end
        if strcmp(PS{L-1,2},'optimal')==1
            cbh = colorbar; a =get(cbh);
            a =  a.Position; set(cbh,'Position',[a(1)+0.13, a(2)-0.05, 0.05, 0.31]);
        end
        cbh = colorbar; a =get(cbh);
        a =  a.Position; set(cbh,'Position',[a(1)+0.13, a(2)-0.12, 0.05, 0.34]);
        cbh.FontSize = 20;
        
    case 'relerr'
        figure,
        lgd_ind = cell(Length_St,1);
        %         LineStyle = {'-.','-','--',':'};
        LineStyle = {'-','-.','--',':'};
%           LineStyle = {'-.','-','--',':'};
        LineColor = [0, 0.4470, 0.7410;      0.8500, 0.3250, 0.0980; 0,0,1;
            0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840];
%         LineColor = [0, 0.4470, 0.7410; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840];

        for L = 1:Length_St
            % Plot relative errors
            if size(PS{L,4}.Enrm) <= maxit1+1
            p(L) = semilogy(1:maxit1, PS{L,4}.Enrm(1:maxit1),LineStyle{mod(L,size(LineStyle,2))+1},'color',LineColor(mod(L,size(LineColor,1))+1,:),'linewidth',3); hold on
            else
                p(L) = semilogy(1:maxit2, PS{L,4}.Enrm(1:maxit2),LineStyle{mod(L,size(LineStyle,2))+1},'color',LineColor(mod(L,size(LineColor,1))+1,:),'linewidth',3); hold on
            end
            % Plot a relative error point at stopping iteration
            %            if PS{L,4}.iterations == maxit
            %             plot(PS{L,4}.iterations,PS{L,4}.Enrm(PS{L,4}.iterations),'o','color',LineColor(mod(L,size(LineColor,1))+1,:),'MarkerFaceColor',LineColor(mod(L,size(LineColor,1))+1,:))
            %            else
            %             plot(PS{L,4}.iterations+1,PS{L,4}.Enrm(PS{L,4}.iterations+1),'o','color',LineColor(mod(L,size(LineColor,1))+1,:),'MarkerFaceColor',LineColor(mod(L,size(LineColor,1))+1,:))
            %            end
            lgd_ind{L} = strcat(PS{L,1},'-',PS{L,2});
            
            if contains(PS{L,1},'mixHyBR') &&(strcmp(PS{L,2},'optimal')==0)
                scatter(PS{L,4}.iterations, PS{L,4}.Enrm(PS{L,4}.iterations),'k','linewidth',6); hold on;
            end
        end
        lgd=legend(p,lgd_ind,'location','northeast');
        lgd.NumColumns=1;
        lgd.FontSize = 10;
        %         a = get(gca,'XTickLabel');
        %         set(gca,'XTickLabel',a,'fontsize',13)
        %         b = get(gca,'YTickLabel');
        %         set(gca,'YTickLabel',b,'fontsize',13)
        set(gca,'fontsize',15)
        ylabel('Relative error','fontsize',15), xlabel('iter','fontsize',15)
        ylim([0 1])
        xlim([1 maxit2])
        %         axis square
        
end


