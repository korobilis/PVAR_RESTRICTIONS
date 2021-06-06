function f=normgen(y,X,beta,sigma2,M)

% PURPOSE: this function computes f(y|X,theta_k) for each regime k=1,...,M+1
% *************************************************************************
% USAGE: f=normgen(y,X,beta,sigma2,M)
% *************************************************************************
% 
% INPUT:
% M is the number of break points, thus M+1 is the number of regimes
% y (scalar)is the dependent var obs at time t
% beta is a (1 x (k+1 x M+1)) vector of parameter under the diff regimes
% sigma2: is a (1 x M+1) vector of variances (can also be a scalar for the constant
%        variance case
% X    is a (1 x k+1) vector containing the data upon which the function evaluates 
%      the conditional density of y (NB: the constant has to be included in the vector)
% OUTPUT:
% f is a (M+1 x 1) vector which contains the evaluation of y under the diff 
%   regimes
% *************************************************************************
% Written by DP on 11/16/03

for i=1:M+1
    f(i) = mean(normpdf(y',(X*beta(:,:,i))',diag(sigma2)));
end

f=f';

