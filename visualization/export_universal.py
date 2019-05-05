import numpy as np
import matplotlib.pyplot as plt

# boundary attack
i = 0
originals = np.load('cifar10/boundary_attack/originals.npz')
originals_d = dict(zip(("data1{}".format(k) for k in originals),
                       (originals[k] for k in originals)))
orig = originals_d['data1img'][i]
print(orig.min(), orig.max())
plt.imsave("figures/universal_orig.pdf", orig)

perturbations = np.load('cifar10/universal_perturbation/DCNet/adv_perts0-999.npy')
adv = np.clip(orig+perturbations[i], 0,1)
print(adv.min(), adv.max())
plt.imsave("figures/universal_adversarial.pdf", adv)

diff = perturbations[i]+0.5
print(diff.min(), diff.max())
plt.imsave('figures/universal_diff.pdf', diff)



