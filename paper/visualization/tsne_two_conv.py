import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import chain, combinations


postfixes = ['0-999', '1000-1999', '2000-2999', '3000-3999', '4000-4999',
             '5000-5999', '6000-6999', '7000-7999', '8000-8999', '9000-9999']

img_size = 32*32*3

names = [
        'DCNet',
        'ConvGood',
        'ConvBaseline',
        ]

perts = {}
for arch_name in names:
    perturbations = [np.load('cifar10/universal_perturbation/'+arch_name+'/adv_perts'+str(i)+'.npy').reshape(-1, 32*32*3) for i in postfixes]
    perts[arch_name] = np.vstack(perturbations)

perturbations = np.vstack([perts[name] for name in names])
perturbations = (perturbations.T/np.linalg.norm(perturbations, axis=1)).T
t = TSNE(n_components=2, random_state=1234, perplexity=42, early_exaggeration=18, learning_rate=200)
output = t.fit_transform(perturbations)
print(output.shape)

plt.figure()

plt.scatter(output[0:100,0], output[0:100,1,],
            marker='.', label='CapsNet')

plt.scatter(output[100:200,0], output[100:200,1,],
            marker='.', label='ConvNet')

plt.scatter(output[200:,0], output[200:,1,],
            marker='.', label='Alternative ConvNet')

plt.legend()
plt.savefig('tsne_two_conv.pdf')
plt.close()
