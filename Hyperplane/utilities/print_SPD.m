function print_SPD(X_spd,Y_spd,Z_spd,size,A_vec,metric,num,fontsize,max_bound,varargin)
    % generate SPD
    if strcmp(metric,'EM')
        theta=varargin{1};
        [X_h,Y_h,Z_h] = gen_hyperplane(num*num*10,max_bound,A_vec,metric,theta(1));
        [X_h2,Y_h2,Z_h2] = gen_hyperplane(num*num*10,max_bound,A_vec,metric,theta(2));
        [X_h3,Y_h3,Z_h3] = gen_hyperplane(num*num*10,max_bound,A_vec,metric,theta(3));
    else
        [X_h,Y_h,Z_h] = gen_hyperplane(num*num*10,max_bound,A_vec{1},metric,varargin);
        [X_h2,Y_h2,Z_h2] = gen_hyperplane(num*num*10,max_bound,A_vec{2},metric,varargin);
        [X_h3,Y_h3,Z_h3] = gen_hyperplane(num*num*10,max_bound,A_vec{3},metric,varargin);
    end
    % Print SPD
    scatter3(X_spd,Y_spd,Z_spd,size,'k','.')
    hold on
    scatter3(X_h,Y_h,Z_h,size,'b','.')
    scatter3(X_h2,Y_h2,Z_h2,size,'r','.')
    scatter3(X_h3,Y_h3,Z_h3,size,'y','.')
    % scatter3(X_h4,Y_h4,Z_h4,'g','.')
    
    xlabel('$x$','interpreter','latex') 
    ylabel('$y$','interpreter','latex') 
    zlabel('$z$','interpreter','latex') 
    metric_name =get_metric_name(metric);
    title(metric_name,'interpreter','latex');
    % legend('Boudary of SPD Manifolds','P=I, A=diag(1,0)','P=I, A=diag(1,1)','P=I, A=diag(1,100)')
    set(gca,'FontSize',fontsize);
    view(-60,30)
    
    % hfig = gcf;
    % figWidth = 7;  % 设置图片宽度 14  7
    % figHeight = 4.3;  % 设置图片高度 8.6 4.3
    % set(hfig,'PaperUnits','centimeters'); % 图片尺寸所用单位
    % set(hfig,'PaperPosition',[0 0 figWidth figHeight]);
    % set(hfig, 'PaperSize', [figWidth+1 figHeight]);
    % % fileout = ['hyperplane.']; % 输出图片的文件名
    % fileout = append('hyperplane_',metric,'.')
    % % print(hfig,[fileout,'tif'],'-r600','-dtiff'); % 设置图片格式、分辨率
    % view(-60,30)
    % print(hfig,[fileout,'pdf'],'-r600','-dpdf'); % 设置图片格式、分辨率
end

function [name]=get_metric_name(metric)
    if strcmp(metric,'LEM') || strcmp(metric,'AIM') || strcmp(metric,'EM')
        name = strcat('$(\alpha,\beta)$-',metric);
    else
        name = strcat('$(\theta)$-',metric);
    end
end 