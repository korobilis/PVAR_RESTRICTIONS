function [atilda,llik0] = dk(y,p,m,t,Qchol,Ht,Qt,Z)

%First draw w as in page 605 of DK
pm = p + m;
wplus = zeros(pm*t,1);
Hchol = chol(Ht);
for i = 1:t
    wplus((i-1)*pm+1:(i-1)*pm+p,1) = Hchol'*randn(p,1);
    wplus((i-1)*pm+p+1:(i-1)*pm+pm,1) = Qchol'*randn(m,1);
end

[yplus,aplus] = recur(Z,wplus,m,p,t);
[what, ahat,llik0] = kalfilt(y,Z,Ht,Qt,m,p,t);
[whatp, ahatp,llik1] = kalfilt(yplus,Z,Ht,Qt,m,p,t);

atilda = ahat - ahatp + aplus;