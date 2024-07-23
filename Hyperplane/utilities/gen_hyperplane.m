function [X,Y,Z] = gen_hyperplane(num,max,A_vec,metric,varargin)
%% corresponding to the hyperplane in sym, satisfying <S',A>=0, with S'= log(S);
%% generate hyperplane 
%% P=I,A=diag(A1,A4) for LEM, satisfying <log(S),A>=0;
%% P=diag{P1,P4},P=I,A=diag(A1,A4) for AIM
%% P=I,A=diag(A1,A4),for \theta-EM;

    X = zeros(num,1);
    Y = zeros(num,1);
    Z = zeros(num,1);
    tmp_spd = zeros(2,2);
    for ith = 1:num
        tmp_spd=gen_2D_spd(max,A_vec,metric,varargin);
        while (~ is_in_cone(tmp_spd,max))
            tmp_spd=gen_2D_spd(max,A_vec,metric,varargin);
        end
        X(ith) = tmp_spd(1);
        Y(ith) = tmp_spd(2);
        Z(ith) = tmp_spd(4);
    end
end

function [spd] = gen_2D_spd(max,A_vec,metric,varargin)
        identity = eye(2);
        if strcmp(metric,'LEM') || strcmp(metric,'AIM') 
            sym = Cal_sym(max,A_vec);
            spd=expm(sym);
        elseif strcmp(metric,'EM')
            theta=varargin{1}{1};
            tmp_spd=zeros(2,2);
            while (is_not_spd(tmp_spd))
                sym = Cal_sym(max,A_vec);
                tmp_spd = sym+identity;
            end
            spd = spd_power(tmp_spd,1\theta);
        elseif strcmp(metric,'BWM')
            tmp_spd=zeros(2,2);
            while (is_not_spd(tmp_spd))
                sym = Cal_sym(max,A_vec);
                tmp_spd = sym+identity;
            end
            spd = tmp_spd*tmp_spd;
        elseif strcmp(metric,'LCM')
            A_vec(1)=0.5*A_vec(1);  A_vec(3)=0.5*A_vec(3);
            sym = Cal_sym(max,A_vec);
            sym(1,2)=0;
            sym(1,1)=exp(sym(1,1));sym(2,2)=exp(sym(2,2));
            spd=sym * sym';
        end   
end

function [sym] = Cal_sym(max,A_vec)
    %% calculating S in <S,A>=0 with A_vec=[A1,A2,A4]
    sym = zeros(2);
    S1 = max*rand();S2 = max*rand();S3 = max*rand();
    A1=A_vec(1); A2=A_vec(2); A4=A_vec(3);
    if A4~=0
        S4=(-1/A4) * (S1*A1+2*S2*A2);
    else
        S4=max*rand();
        if A1~=0
            S1=((-2*A2)/A1) * S2 ;
        else
            S2=0;
        end
    end
    sym(1) = S1;sym(2)=S2;sym(3)=S2;sym(4) = S4;
end
        
function result = is_in_cone(X,max)
    if abs(X(1))<=max && abs(X(4))<=max && abs(X(2))<=max
        result = true;
    else
        result = false;
    end
end

function [spd] = gen_2dim_spd_t2(max)
    sym = zeros(2,2);
    x_1 = max*rand();
    x_2 = max*rand();
    sym(1) = x_1;
    sym(4) = x_2;
    spd=expm(sym);
end

function [results] = is_not_spd(X)
    [U,S] = eig(X);
    results = any(diag(S) <= 0);
end
