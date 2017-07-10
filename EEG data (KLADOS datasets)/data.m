importfile('Pure_Data');
importfile('Contaminated_Data');


%%
subplot(2,1,1)
plot(sim10_resampled(1,:));
ylabel('Power')
set(gca,'XTick',[]);
subplot(2,1,2)
plot(sim10_con(1,:));
xlabel('Sample')
ylabel('Power')