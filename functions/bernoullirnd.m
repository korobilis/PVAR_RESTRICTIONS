function [x] = bernoullirnd(p)
%-------------------------------------------------------------------
% Generate a scalar x from Bernoulli distribution
%-------------------------------------------------------------------
% x is a 0-1 variable following the distribution:
%                    
%         prob(x) = [(p)^x][(1-p)^(1-x)]
%
%-------------------------------------------------------------------
%  INPUTS:
%   n     - The number of variates you want to generate
%   p     - Associated probability
%
%  OUTPUT:
%   x     - [1x1] Bernoulli variate
%-------------------------------------------------------------------

   
u=rand;
if u<p   
    x = 1;
else
    x = 0;
end
