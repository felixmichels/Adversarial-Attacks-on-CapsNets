import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


postfixes = ['0-2603', '2604-5207', '5208-7810', '7811-10413', '10414-13016',
             '13017-15619', '15620-18222', '18223-20825', '20826-23428', '23429-26031']
caps_perturbations = [np.load('svhn/universal_perturbation/CapsNetVariant/adv_perts'+str(i)+'.npy').reshape(-1, 32*32*3) for i in postfixes]

caps_perturbations= np.vstack(caps_perturbations)
print(caps_perturbations.shape)


cnn_perturbations = [np.load('svhn/universal_perturbation/ConvBaseline/adv_perts'+str(i)+'.npy').reshape(-1, 32*32*3) for i in postfixes]

cnn_perturbations= np.vstack(cnn_perturbations)
print(cnn_perturbations.shape)

perturbations = np.vstack([caps_perturbations, cnn_perturbations])
print(perturbations.shape)

perturbations = (perturbations.T/np.linalg.norm(perturbations, axis=1)).T

t = TSNE(n_components=2, random_state=1234)
output = t.fit_transform(perturbations)
print(output.shape)

# Rotate to match image of cifar10 tsne
output *= -1

fig, ax = plt.subplots()
ax.scatter(output[0:100,0], output[0:100,1],
            c=np.repeat(np.arange(0,10), 10), marker='.')

ax.scatter(output[100:200,0], output[100:200,1],
            c=np.repeat(np.arange(0,10), 10), marker='^')

plt.savefig('tsne_svhn.pdf')
