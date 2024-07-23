function [X_power] = spd_power(X,theta)
%% for SPD matrix
    if det(X)<=0
        error("Wrong SPD")
    end
    [U, S, V] = svd(X);
    S_power = S.^theta;
    X_power = U * S_power * U';
end