import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from PIL import Image
import sys


def fcm(X, C, m, max_iter=100, error=1e-5, init=None):
    # Initialize membership matrix using K-means algorithm
    if init == 'kmeans':
        kmeans = KMeans(n_clusters=C, n_init="auto")
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        D = cdist(X, centroids)
        U = 1 / (D ** (2 / (m - 1)) + 1e-8)
        U = U / np.sum(U, axis=1, keepdims=True)
    else:
        # Initialize membership matrix randomly
        N = X.shape[0]
        U = np.random.rand(N, C)
        U = U / np.sum(U, axis=1, keepdims=True)
        centroids = None

    for i in range(max_iter):
        # Compute centroids
        centroids = U.T @ X / np.sum(U, axis=0, keepdims=True).T

        # Compute distance matrix
        D = cdist(X, centroids)

        # Compute membership matrix
        U_new = 1 / (D ** (2 / (m - 1)) + 1e-8)
        U_new = U_new / np.sum(U_new, axis=1, keepdims=True)

        # Check for convergence
        if np.linalg.norm(U_new - U) < error:
            break

        # Update membership matrix
        U = U_new

    return centroids, U

if sys.version_info.major == 3:
    python_command = "python3"
else:
    python_command = "python"

if(len(sys.argv)!=4):
    print(f"Error: Usage {python_command} {sys.argv[0]} input_image_name C_min C_max")
    exit(1)

imgName = sys.argv[1]
C_min = int(sys.argv[2])
C_max = int(sys.argv[3])

# Load the sample image
image1 = io.imread(imgName)
image = color.rgb2gray(image1)

# Normalize the image values
scaler = MinMaxScaler()
image = scaler.fit_transform(image)

# Flatten the image into a 2D array
X = image.reshape(-1, 1)

# Apply FCM clustering algorithm

Clusters = range(C_min, C_max+1)
scores = []
outputs = []
Labels = []

for k in Clusters:

    C = k  # Number of clusters
    m = 2  # Fuzziness coefficient
    max_iter = 100  # Maximum number of iterations
    error = 1e-5  # Convergence threshold
    centroids, U = fcm(X, C, m, max_iter, error, init='kmeans')

    # Compute Silhouette score
    labels = np.argmax(U, axis=1)
    Labels.append(labels)
    # score = silhouette_score(X, labels)
    # score = calinski_harabasz_score(X, labels)
    score = davies_bouldin_score(X, labels)
    scores.append(score)
    # Reshape the labels and centroids back to the original image shape
    labels = labels.reshape(image.shape)
    centroids = centroids.reshape((C,))

    plt.imshow(labels, cmap='gray')
    plt.title(f'Segmented image (C={C}, score={score:.4f})')
    for c in centroids:
        plt.axhline(c, color='white')
    outfileName = 'output_C' + str(k) + ".jpeg"
    outputs.append(outfileName)
    plt.savefig(outfileName)
    print(f"Iteration corresponding to C={k} done")

MaxScore = max(scores)
index_maxScore = scores.index(MaxScore)
C_max = Clusters[index_maxScore]
output = outputs[index_maxScore]
label = Labels[index_maxScore]
label = label.reshape(image.shape)
# Read image
img = Image.open(output)
 
# Output Images
img.show()

print(f"Max Score corresponds to image with Number of clusters = {C_max} with score of {MaxScore}")
#Plot the segmented image
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

ax0.imshow(image1, cmap='gray')
ax0.set_title('Original image')

ax1.imshow(label, cmap='gray')
ax1.set_title(f"Segmented Image for C = {C_max} with Max Score")

plt.savefig("Original_BestSegmented.jpeg")

img = Image.open("Original_BestSegmented.jpeg")
img.show()