import numpy as np
import matplotlib.pyplot as plt

# boundary attack
i = 7
originals = np.load('cifar10/carlini_wagner/originals.npz')
originals_d = dict(zip(("data1{}".format(k) for k in originals),
                       (originals[k] for k in originals)))
orig = originals_d['data1img'][i]
print(orig.min(), orig.max())
plt.imsave("figures/carlini_wagner_orig.pdf", orig)

adversarial = np.load('cifar10/carlini_wagner/DCNet/adv_images.npy')
adv = adversarial[i]
print(adv.min(), adv.max())
plt.imsave("figures/carlini_wagner_adversarial.pdf", adv)

diff = ((orig-adv)+0.5)
print(diff.min(), diff.max())
plt.imsave('figures/carlini_wagner_diff.pdf', diff)



