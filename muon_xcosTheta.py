# calculate the lepton polar angle of muons and the reduced lepton energy x
# we use the reconstructed muon energy as E_l and (for now) resort to
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
muon_px = np.array(ak.flatten(df["muon_px"]))
muon_py = np.array(ak.flatten(df["muon_py"]))
muon_pz = np.array(ak.flatten(df["muon_pz"]))
muon_energy = np.array(ak.flatten(df["muon_energy"]))
muon_charge = np.array(ak.flatten(df["muon_charge"]))

#select positively and negatively charges muons
mask = muon_charge==-1 #True for -1 and False for +1

#calculate x and cos(Theta) for positive and negative muons respectively
m_t = 173 #top mass in GeV
muon_pos_px, muon_pos_py, muon_pos_pz, muon_pos_E = muon_px[~mask], muon_py[~mask], muon_pz[~mask], muon_energy[~mask]
muon_neg_px, muon_neg_py, muon_neg_pz, muon_neg_E = muon_px[mask], muon_py[mask], muon_pz[mask], muon_energy[mask]
muon_pos_pt = np.sqrt(muon_pos_px**2 + muon_pos_py**2)
muon_neg_pt = np.sqrt(muon_neg_px**2 + muon_neg_py**2)
theta_pos = np.arctan(muon_pos_pt/muon_pos_pz)+np.pi/2 #lepton polar angle with Theta=0 in +z and -pi in -z direction
theta_neg = np.arctan(muon_neg_pt/muon_neg_pz)+np.pi/2
x_pos = 2*muon_pos_E/m_t #reduced lepton energy where the fraction of the top velocity have been approx to be 1
x_neg = 2*muon_neg_E/m_t

#histogram the data
#for the positively charged muons
fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')


hist, xedges, yedges = np.histogram2d(x_pos, np.cos(theta_pos), bins=(50,50))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
plt.title(r"$S^{0}(x,cos(\Theta)$ for $\mu^{+}$")
plt.xlabel("x")
plt.ylabel(r"$cos(\Theta)$")
plt.savefig("x_cosTheta_muon_pos.png")

#for negatively charged muons
fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(x_neg, np.cos(theta_neg), bins=(50,50))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
plt.title(r"$S^{0}(x,cos(\Theta)$ for $\mu^{-}")
plt.xlabel("x")
plt.ylabel(r"$cos(\Theta)$")
plt.savefig("x_cosTheta_muon_neg.png")



