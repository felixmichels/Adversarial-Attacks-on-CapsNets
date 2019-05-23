import numpy as np
import matplotlib.pyplot as plt

# boundary attack
i = 15
originals = np.load('cifar10/boundary_attack/originals.npz')
originals_d = dict(zip(("data1{}".format(k) for k in originals),
                       (originals[k] for k in originals)))
orig = originals_d['data1img'][i]
print(orig.min(), orig.max())
plt.imsave("figures/boundary_orig.pdf", orig)

adversarial = np.load('cifar10/boundary_attack/DCNet/adv_images.npy')
adv = adversarial[i]
print(adv.min(), adv.max())
plt.imsave("figures/boundary_adversarial.pdf", adv)

diff = ((orig-adv)+0.5)
print(diff.min(), diff.max())
plt.imsave('figures/boundary_diff.pdf', diff)



