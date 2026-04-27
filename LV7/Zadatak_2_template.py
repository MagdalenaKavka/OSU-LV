import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

for p in range(1,7):
    # ucitaj sliku
    img = Image.imread(f"LV7/test_{p}.jpg")
    
    if p == 4:
        img = img[:, :, :3]
    else:
        # pretvori vrijednosti elemenata slike u raspon 0 do 1
        img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    fig, axes = plt.subplots(3, 2, figsize=(12,10))
    fig.suptitle(f'Test {p} slika')
    axes = axes.flatten()

    # prikazi originalnu sliku
    axes[0].set_title("Originalna slika")
    axes[0].imshow(img)

    k_clusters = [2,3,5,10,15]
    for i in range(5):
        km = KMeans(n_clusters=k_clusters[i], init='k-means++', n_init=5)
        km.fit(img_array)
        labels = km.predict(img_array)

        img_array_aprox = km.cluster_centers_[labels]
        img_aprox = np.reshape(img_array_aprox, (w, h, d))

        axes[i+1].set_title(f"Kvantizirana slika k={km.n_clusters}")
        axes[i+1].imshow(img_aprox)

    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12,10))
axes = axes.flatten()

for p in range(1,7):
    img = Image.imread(f"imgs\\test_{p}.jpg")
    
    if p == 4:
        img = img[:, :, :3]
    else:
        img = img.astype(np.float64) / 255

    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    inertias = []
    k_values = range(1, 11)

    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=5)
        km.fit(img_array)
        inertias.append(km.inertia_)

    axes[p-1].plot(k_values, inertias, 'bo-')
    axes[p-1].set_xlabel('K')
    axes[p-1].set_ylabel('Inertia (J)')
    axes[p-1].set_title(f'Test {p} slika')

plt.tight_layout()
plt.show()

optimal_k = [3, 2, 2, 2, 4, 2]
for p in range(1,7):
    img = Image.imread(f"imgs\\test_{p}.jpg")
    
    if p == 4:
        img = img[:, :, :3]
    else:
        img = img.astype(np.float64) / 255

    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    km = KMeans(n_clusters=optimal_k[p-1], init='k-means++', n_init=5)
    km.fit(img_array)
    labels = km.predict(img_array)
    labels_2d = labels.reshape(w, h)
    if optimal_k[p-1] <= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    elif optimal_k[p-1] == 3:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    else:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()
    fig.suptitle(f'Test {p} slika, optimalni k: {optimal_k[p-1]}')

    img_array_aprox = km.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[0].set_title(f"Slika u {optimal_k[p-1]} grupe")
    axes[0].imshow(img_aprox)

    for k in range(optimal_k[p-1]):
        binary = (labels_2d == k).astype(float)  # 1 gdje je grupa k, inace 0
        axes[k+1].imshow(binary, cmap='gray')
        axes[k+1].set_title(f'Grupa {k}')

    plt.tight_layout()
    plt.show()