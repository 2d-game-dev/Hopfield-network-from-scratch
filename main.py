import numpy as np
import matplotlib.pyplot as plt

pattern_8 = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0]
]

pattern_8 = np.array(pattern_8)  
pattern = np.where(pattern_8 == 1, 1, -1)#Convert {0,1} → {-1,+1}
pattern = pattern.flatten()


#Hebbian learning for a single pattern
W= np.outer(pattern, pattern)
np.fill_diagonal(W, 0)


def add_noise(state, num_flips = 6):
    noisy = state.copy()
    num_flips = min(num_flips, len(state)) 
    index = np.random.choice(len(state), num_flips, replace=False)
    noisy[index] *= -1 #Flip the selected bits
    return noisy

noisy_pattern = add_noise(pattern, num_flips=6)


#Hopfield recall(async)

def hopfield_recall(state, W, max_iters=30):
    s = state.copy()
    for _ in range(max_iters):
        prev = s.copy()
        for i in range(len(s)):
            s[i] = 1 if np.dot(W[i], s) >= 0 else -1
        if np.array_equal(s, prev):
            break
    return s

recovered_pattern = hopfield_recall(noisy_pattern, W) 

#Visualization

def show(img, title):
    plt.imshow(img.reshape(8, 8), cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1); show(pattern, "Original")
plt.subplot(1, 3, 2); show(noisy_pattern, "Noisy")
plt.subplot(1, 3, 3); show(recovered_pattern, "Recovered")
plt.show()