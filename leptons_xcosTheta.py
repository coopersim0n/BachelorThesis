import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def arrays(particle):
	home_dir = os.path.expanduser("~")
	names = ["{}_px.npy", "{}_py.npy", "{}_pz.npy", "{}_charge.npy", "{}_energy.npy"]
	formatted_names = [name.format(particle) for name in names]
	path_to_array = []
	for i in formatted_names:
		path_to_array.append(os.path.join(home_dir,"arrays",i))
	output = []
	for i in range(len(path_to_array)):
		output.append(np.load(path_to_array[i]))
	return output

electron_px, electron_py, electron_pz, electron_charge, electron_energy = arrays("electron")
muon_px, muon_py, muon_pz, muon_charge, muon_energy = arrays("muon")

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
#joining lepton arrays
lepton_px = np.concatenate((electron_px,muon_px))
lepton_py = np.concatenate((electron_py,muon_py))
lepton_pz = np.concatenate((electron_pz,muon_pz))
lepton_energy = np.concatenate((electron_energy,muon_energy))
lepton_charge = np.concatenate((electron_charge,muon_charge))

#select positively and negatively charged leptons
mask = lepton_charge==-1 #True for -1 and False for +1

#calculate x and cos(Theta) for positive and negative leptons respectively
print("---calculating x and Theta---")
m_t = 173 #top mass in GeV
lepton_pos_px, lepton_pos_py, lepton_pos_pz, lepton_pos_E = lepton_px[~mask], lepton_py[~mask],lepton_pz[~mask], lepton_energy[~mask]
lepton_neg_px, lepton_neg_py, lepton_neg_pz, lepton_neg_E = lepton_px[mask], lepton_py[mask], lepton_pz[mask], lepton_energy[mask]
lepton_pos_pt = np.sqrt(lepton_pos_px**2 + lepton_pos_py**2)
lepton_neg_pt = np.sqrt(lepton_neg_px**2 + lepton_neg_py**2)
#calculate lepton polar angle with Theta=0 in +z and -pi in -z direction
num_entries_pos = len(lepton_pos_pt)
num_entries_neg = len(lepton_neg_pt)
Theta_pos = np.zeros(num_entries_pos)
Theta_neg = np.zeros(num_entries_neg)

def theta(pt,pz):
        if pz>0:
                return np.arctan(pt/pz)
        elif pz<0:
                return  np.arctan(pt/pz)+np.pi

for i in range(num_entries_pos):
        Theta_pos[i] = theta(lepton_pos_pt[i],lepton_pos_pz[i])
for i in range(num_entries_neg):
        Theta_neg[i] = theta(lepton_neg_pt[i],lepton_neg_pz[i])

x_pos = 2*lepton_pos_E/m_t #reduced lepton energy where the fraction of the top velocity have been approx to be 1
x_neg = 2*lepton_neg_E/m_t

hist, xedges, yedges = np.histogram2d(x_neg,np.cos(Theta_neg), bins=(100,100))
xpos, ypos = np.meshgrid(yedges[:-1]+yedges[1:], xedges[:-1]+xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
xpos = xpos/2.
ypos = ypos/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]

#ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax2.plot_wireframe(xpos,ypos,hist, rstride=3, cstride=3)
ax2.set_title(r"$S^{0}(x,cos(\Theta))$ for $l^{-}$")
ax2.set_xlabel(r"$cos(\Theta)$")
ax2.set_ylabel("x")
ax2.set_xticks([-1,0,1])
ax2.set_yticks([0.2,0.4,0.6,0.8,1.])
plt.savefig("FCCee_leptons_xcosTheta.png",dpi=300)

	
