from matplotlib.pyplot import figure, show
from numpy import arange, sin, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#t = arange(0.0, 1.0, 0.01)
#noise1 = np.random.normal(0, 0.2, 100)
#noise2 = np.random.normal(0, 0.4, 100)
#fig = figure(1)
#
#ax1 = fig.add_subplot(211)
#ax1.plot(t, sin(4*pi*t)+noise1)
#ax1.plot(t, sin(8.7*pi*t+0.2*pi)+noise2, '.')
#ax1.plot(t, sin(8.7*pi*t+0.2*pi)+noise2, 'g-')
#
#ax1.text(0.7, -2.3, r'$\rho = {:.2f}$'.format(np.corrcoef( sin(4*pi*t)+noise1, sin(8.7*pi*t+0.2*pi)+noise2)[1][0]), fontsize=15)
#ax1.set_ylim((-3, 2))
#ax1.set_ylabel('weak correlation')
#
#
#ax2 = fig.add_subplot(212)
#ax2.plot(t, sin(4*pi*t)+noise1)
#ax2.plot(t, sin(4.2*pi*t)+noise2, '.')
#ax2.plot(t, sin(4.2*pi*t)+noise2,  'g-')
#ax2.text(0.7, -2.3, r'$\rho = {:.2f}$'.format(np.corrcoef( sin(4*pi*t)+noise1, sin(4.2*pi*t)+noise2)[1][0]), fontsize=15)
#ax2.set_ylim((-3, 2))
#ax2.set_ylabel('strong correlation')
#l = ax2.set_xlabel('Time (s)')
#
#fig.savefig('corr.png',dpi=300)
#show()



#t = arange(0.0, 1.0, 0.01)
#noise1 = np.random.normal(0, 1.2, 100)
#noise2 = np.random.normal(0, 0.4, 100)
#sign = sin(4*pi*t)+noise1
#artefact = sin(8.7*pi*t+0.2*pi)+noise2+1.2*signal.square(8.7 * pi * t+0.2*pi)
#fig = figure(figsize=(6,6))
#
#ax1 = fig.add_subplot(411)
#ax1.plot(t, sign, 'g')
#
#
#ax1.set_ylim((-8, 6))
#ax1.set_xticklabels([])
#l = ax1.set_ylabel('$B^s(t)$')
#l.set_fontsize('large')
#
#ax2 = fig.add_subplot(412)
#ax2.plot(t,artefact, 'r')
#ax2.set_xticklabels([])
#ax2.set_ylim((-8, 6))
#l = ax2.set_ylabel('$O^s(t)$')
#l.set_fontsize('large')
#
#
#
#
#ax3 = fig.add_subplot(413)
#ax3.plot(t, sign+artefact, 'b')
#ax3.text(0.45, -6.3, r'$SNR_X = {:.2f}$'.format((0.01*sum(sign**2))/(0.01*(sum(artefact**2)))), fontsize=15)
#ax3.set_ylim((-8, 6))
#ax3.set_xticklabels([])
#l = ax3.set_ylabel('$X^s(t)$')
#l.set_fontsize('large')
#
#corr = sign+artefact - 0.9*artefact + noise2
#ax4 = fig.add_subplot(414)
#ax4.plot(t, corr)
#ax4.set_ylim((-8, 6))
#ax4.text(0.45, -6.3, r'$SNR_C = {:.2f}$'.format((0.01*sum(sign**2))/(0.01*(sum((corr-sign)**2)))), fontsize=15)
#l = ax4.set_ylabel('$C^s(t)$')
#l.set_fontsize('large')
#ax1.set_xticklabels([])
##fig.savefig('signals.png',dpi=300)
#
#l = ax4.set_xlabel('Time (s)')
#show()
#
#t = arange(0.0, 1.0, 0.01)
#noise1 = np.random.normal(0, 0.9, 100)
#noise2 = np.random.normal(0, 0.9, 100)
#noise3 = np.random.normal(0,0.9,100)
#sign1 = 2.5*sin(6*pi*t)+noise1
#sign2 = 2.5*sin(12*pi*t)+noise3
#artefact = 3.0*signal.square(8.7 * pi * t+0.2*pi)+noise2
#
#f, axarr = plt.subplots(3, 4, figsize=(16,4),sharex=True, sharey=True)
#plt.subplots_adjust(wspace=0.5)
#axarr[0, 0].plot(t, sign1,'g')
#
#axarr[2, 0].plot(t, sign2,'g')
#
#axarr[1, 0].plot(t,artefact, 'r')
#
#axarr[0, 1].axis('off')
#axarr[2, 1].axis('off')
#axarr[0, 3].axis('off')
#axarr[2, 3].axis('off')
##axarr[0, 1].set_title('Axis [0,1]')
#axarr[1, 1].plot(t, 0.9*sign1+0.8*sign2+1.2*artefact)
#axarr[1, 1].set_title('')
#
#axarr[0, 2].plot(t, sign1,'g')
#
#axarr[1, 2].plot(t, sign2,'g')
#
#axarr[2, 2].plot(t,artefact, 'r')
#axarr[1, 3].plot(t, sign1+sign2+0.01*artefact)
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
#plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 3]], visible=False)
#f.savefig('mixingunmixing.png',dpi=300)





t = arange(0.0, 1.0, 0.01)
noise1 = np.random.normal(0, 2.5, 100)
noise2 = np.random.normal(0, 0.9, 100)
noise3 = np.random.normal(0,0.9,100)
sign1 = 2.5*sin(4*pi*t)+noise1
sign2 = 2.5*sin(12*pi*t)+noise3
artefact = 5.0*signal.square(8.7 * pi * t+0.2*pi)+noise2

f, axarr = plt.subplots(2, 4, figsize=(16,2.6),sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.5)
axarr[0, 0].plot(t, sign1,'g')
l = axarr[0, 0].set_ylabel('$B^s(t)$')
l.set_fontsize('large')

axarr[0, 2].axis('off')

axarr[1, 0].plot(t,artefact, 'r')
l = axarr[1, 0].set_ylabel('$O^s(t)$')
l.set_fontsize('large')
axarr[0, 1].axis('off')
axarr[1,2].axis('off')

axarr[1, 1].plot(t, 0.9*sign1+1.2*artefact)
l = axarr[1, 1].set_ylabel('$X^s(t)$')
l.set_fontsize('large')
axarr[1, 1].set_title('')
axarr[0, 3].axis('off')
axarr[1, 3].plot(t, sign1+0.01*artefact+2*noise2)
l = axarr[1, 3].set_ylabel('$C^s(t)$')
l.set_fontsize('large')
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
#plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
#plt.setp([a.get_xticklabels() for a in axarr[3, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
f.savefig('simulateddata.png',dpi=300)