# calculate the lepton polar angle of muons and the reduced lepton energy x
# we use the reconstructed electron energy as E_l and (for now) resort to
# approximating m_t with the actual top mass of 173 GeV and the square with
# the top velocity as 1

#import packages
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak

#import dataframe
df = pd.read_pickle("~/FCC_ee.pkl")
electron_px = np.array(ak.flatten(df["electron_px"]))
electron_py = np.array(ak.flatten(df["electron_py"]))
electron_pz = np.array(ak.flatten(df["electron_pz"]))
electron_energy = np.array(ak.flatten(df["electron_energy"]))
electron_charge = np.array(ak.flatten(df["electron_charge"]))
print("---calculating x and Theta---")
#select positively and negatively charges muons
mask = electron_charge==-1 #True for -1 and False for +1

#calculate x and cos(Theta) for positive and negative muons respectively
m_t = 173 #top mass in GeV
positron_px, positron_py, positron_pz, positron_E = electron_px[~mask], electron_py[~mask],electron_pz[~mask], electron_energy[~mask]
electron_px, electron_py, electron_pz, electron_E = electron_px[mask], electron_py[mask], electron_pz[mask], electron_energy[mask]
positron_pt = np.sqrt(positron_px**2 + positron_py**2)
electron_pt = np.sqrt(electron_px**2 + electron_py**2)
#calculate lepton polar angle with Theta=0 in +z and -pi in -z direction
num_entries_pos = len(positron_pt)
num_entries_el = len(electron_pt)
Theta_pos = np.zeros(num_entries_pos)
Theta_el = np.zeros(num_entries_el)

def theta(pt,pz):
        if pz>0:
                return np.arctan(pt/pz)
        elif pz<0:
                return  np.arctan(pt/pz)+np.pi

for i in range(num_entries_pos):
        Theta_pos[i] = theta(positron_pt[i],positron_pz[i])
for i in range(num_entries_el):
	Theta_el[i] = theta(electron_pt[i],electron_pz[i])

x_pos = 2*positron_E/m_t #reduced lepton energy where the fraction of the top velocity have been approx to be 1
x_el = 2*electron_E/m_t
print("---creating the histograms---")
#histogram the data
#for positrons
fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax1 = fig.add_subplot(121, projection='3d')


hist, xedges, yedges = np.histogram2d(np.cos(Theta_pos), x_pos, bins=(50,50))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax1.set_title(r"$S^{0}(x,cos(\Theta))$ for $e^{+}$")
ax1.set_xlabel(r"$cos(\Theta)$")
ax1.set_ylabel("x")
ax1.set_xticks([-1,0,1])
ax1.set_yticks([0.2,0.4,0.6,0.8,1.])

#for electrons
ax2 = fig.add_subplot(122, projection='3d')

hist, xedges, yedges = np.histogram2d(np.cos(Theta_el), x_el, bins=(50,50))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax2.set_title(r"$S^{0}(x,cos(\Theta)$ for $e^{-}$")
ax2.set_xlabel(r"$cos(\Theta)$")
ax2.set_ylabel("x")
ax2.set_xticks([-1,0,1])
ax2.set_yticks([0.2,0.4,0.6,0.8,1.])
plt.savefig("electrons_x_cosTheta.png")

#plot each distribution on its own
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(x_el,histtype="step", bins=100)
axs[0].set_xlabel("x")
axs[0].set_xticks([0.2,0.4,0.6,0.8,1])
axs[0].set_ylabel("events")
axs[1].hist(np.cos(Theta_el),histtype="step", bins=100)
axs[1].set_xlabel(r"$cos(\Theta)$")
axs[1].set_ylabel("events")
axs[1].set_xticks([-1,0,1])
fig.suptitle(r"x and $cos(\Theta)$ for $e^{-}$")
plt.savefig("1d_hist_electrons.png")


