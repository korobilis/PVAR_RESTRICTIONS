function [ydraw,alpha] = recur(Z,wdraw,m,p,t)
%Now get implied draw of y
alpha = zeros(m,t+1);
pm = p + m;
ydraw = zeros(p,t);
for i = 1:t
    ztemp = Z((i-1)*p+1:i*p,:);
    ydraw(:,i) = ztemp*alpha(:,i) + wdraw((i-1)*pm+1:(i-1)*pm+p,1);
    alpha(:,i+1) = alpha(:,i) + wdraw((i-1)*pm+p+1:i*pm,1) ;
end