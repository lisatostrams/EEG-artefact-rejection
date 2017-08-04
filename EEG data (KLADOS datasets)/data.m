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


%%
   %% Time specifications:
   Fs = 5801;                   % samples per second
   dt = 1/Fs;                   % seconds per sample
   StopTime = 1;             % seconds
   t = (0:dt:StopTime-dt)';     % seconds
   %% Sine wave:
   Fc = 10;                     % hertz
   x = cos(2*pi*Fc*t);
   % Plot the signal versus time:
   figure;
   plot(t,x+0.05.*sim10_resampled(1,:)');
   xlabel('time (in seconds)');
   title('Signal versus Time');
   zoom xon;