load('runtimes.mat') % load file created by ./runtimes_evalutation.py
cut = 9;

times = runtimes_matrix(end,:);
shorttimes = times(1:cut);

wwl = runtimes_matrix(1,1:cut);
wwl_e = runtimes_matrix(2,1:cut);
sw = runtimes_matrix(3,1:end);
rpwq_sq = runtimes_matrix(4,1:cut);
rpwq = runtimes_matrix(5,1:end);
pwq = runtimes_matrix(6,1:end);

linewidth = 1.5;%1.3; %2
legendFontSize = 26; % 18

loglog(shorttimes,wwl,'linewidth',linewidth)
hold on
loglog(shorttimes,wwl_e,'linewidth',linewidth)
loglog(times,sw,'linewidth',linewidth)
loglog(shorttimes,rpwq_sq,'linewidth',linewidth)
loglog(times,rpwq,'linewidth',linewidth)
loglog(times,pwq,'linewidth',linewidth)

legend({'$\mathcal{W}_2$ POT','$\mathcal{W}_2^e$ ($\gamma = 100)$ POT','$\mathcal{SW}_2$ POT','$\mathcal{RPW}_2$ Quad. Numpy','$\mathcal{RPW}_2$ Numpy', '$\mathcal{PW}_2$ Numpy'}, 'interpreter','latex',...
    'fontSize',legendFontSize)

xlim([10^1 10^8])
ax = gca;
ax.FontSize = 16; % 16
xlabel('Number of samples','interpreter','latex','fontSize',30) %,26
ylabel('Computing times (s)','interpreter','latex','fontSize',30) %;26
grid on

