%Calculate impulse responses

function [IRF]=irfvar(A,A0inv,p,h)

q=size(A0inv,1);
J=[eye(q,q) zeros(q,q*(p-1))];
IRF=reshape(J*A^0*J'*A0inv,q^2,1);

for i =1:h
    IRF=([IRF reshape(J*A^i*J'*A0inv,q^2,1)]);
end
