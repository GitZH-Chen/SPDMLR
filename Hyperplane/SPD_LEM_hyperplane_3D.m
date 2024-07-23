clear
clc
confPath
rng(0);

dim=2;
num=30;
max_bound = 3;
size = 10;
fontsize = 14;
Z = zeros(num,num);
mask = find(tril(ones(2,2))>0);
SPD_X=zeros(num,num);
SPD_Y=zeros(num,num);
SPD_Z=zeros(num,num);

[tmp_X,tmp_Z] = meshgrid(linspace(0,max_bound,num),linspace(0,max_bound,num));
tmp_Y = sqrt(tmp_X.*tmp_Z);
X_spd=[tmp_X,tmp_X];
Z_spd=[tmp_Z,tmp_Z];
Y_spd=[tmp_Y,-tmp_Y];

metric={'LEM','LCM'};
% % LEM
ith=1;
% A_vec={[2,0.5,1],[1,1,2],[1,1,10]};
A_vec={[10,0,0],[1,1,0],[0,0,10]};
print_SPD(X_spd,Y_spd,Z_spd,size,A_vec,metric{ith},num,fontsize,max_bound)

hfig = gcf;
figWidth = 16;  % Width of one column in inches (adjust as needed)
figHeight = 10; % Height of the figure in inches (adjust as needed)
set(hfig, 'Units', 'centimeters','Position', [1, 1, figWidth, figHeight]);
% set(hfig, 'PaperSize', [figWidth+1 figHeight])

% figWidth = 14;  % 设置图片宽度 14  7
% figHeight = 8.6;  % 设置图片高度 8.6 4.3
% set(hfig,'PaperUnits','centimeters'); % 图片尺寸所用单位
% set(hfig,'PaperPosition',[0 0 figWidth figHeight]);
% set(hfig, 'PaperSize', [figWidth+1 figHeight]);
% fileout = ['hyperplane.']; % 输出图片的文件名
fileout = append('hyperplanes_single2','.');
% print(hfig,[fileout,'tif'],'-r600','-dtiff'); % 设置图片格式、分辨率
axis off;
title('');
print(hfig,[fileout,'jpg'],'-r600','-djpeg'); % 设置图片格式、分辨率
% print(hfig,[fileout,'pdf'],'-r600','-dpdf'); % 设置图片格式、分辨率

function [x,y,z] = geodesic(vx,vy,vz,t)
%% geodesic starting at I, with velolcity of v (correspongding velolcity in sym)
    v=[vx,vy;vy,vz];
    p=[0.5819,0.8811; 0.8811,-0.5819];
%     p=[2.0329,3.0810; 3.0810,-2.0326];
    x=zeros(length(t),1);
    y=zeros(length(t),1);
    z=zeros(length(t),1);
    for ith = 1:length(t)
        tmp = expm(p+v*t(ith));
        x(ith)=tmp(1);
        y(ith)=tmp(2);
        z(ith)=tmp(4);
    end
end