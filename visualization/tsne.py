import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


postfixes = ['0-999', '1000-1999', '2000-2999', '3000-3999', '4000-4999',
             '5000-5999', '6000-6999', '7000-7999', '8000-8999', '9000-9999']
caps_perturbations = [np.load('cifar10/universal_perturbation/DCNet/adv_perts'+str(i)+'.npy').reshape(-1, 32*32*3) for i in postfixes]

caps_perturbations= np.vstack(caps_perturbations)
print(caps_perturbations.shape)


cnn_perturbations = [np.load('cifar10/universal_perturbation/ConvGood/adv_perts'+str(i)+'.npy').reshape(-1, 32*32*3) for i in postfixes]

cnn_perturbations= np.vstack(cnn_perturbations)
print(cnn_perturbations.shape)

perturbations = np.vstack([caps_perturbations, cnn_perturbations])
print(perturbations.shape)

perturbations = (perturbations.T/np.linalg.norm(perturbations, axis=1)).T

t = TSNE(n_components=2, random_state=1234)
output = t.fit_transform(perturbations)
print(output.shape)

col = np.hstack([np.repeat(np.arange(0,10), 10),
                 np.repeat(np.arange(0,10), 10)])
print(col.shape)

fig, ax = plt.subplots()
ax.scatter(output[0:100,0], output[0:100,1],
            c=np.repeat(np.arange(0,10), 10), marker='.')

ax.scatter(output[100:200,0], output[100:200,1],
            c=np.repeat(np.arange(0,10), 10), marker='^')

plt.savefig('tsne.pdf')
