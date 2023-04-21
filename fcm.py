import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, io
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score
from PIL import Image
import sys
#Import relevant libraries

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
#Check Sys args
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

    # Compute Davies Bouldin score
    labels = np.argmax(U, axis=1)
    Labels.append(labels)
    score = davies_bouldin_score(X, labels)
    scores.append(score)
    # Reshape the labels and centroids back to the original image shape
    labels = labels.reshape(image.shape)
    centroids = centroids.reshape((C,))
    #plot and save the clustered image
    plt.imshow(labels, cmap='gray')
    plt.title(f'Clustered image (C={C}, score={score:.4f})')
    for c in centroids:
        plt.axhline(c, color='white')
    outfileName = 'output_C' + str(k) + ".jpeg"
    outputs.append(outfileName)
    plt.savefig(outfileName, dpi=300)
    print(f"Iteration corresponding to C={k} done")
    plt.close()

MinScore = min(scores) #Minimum Davies Boulding Score
index_minScore = scores.index(MinScore) #Corresponding index of MinScore
C_opt = Clusters[index_minScore] #Optimal C value which corresponds to the minimum Davies Boulding Score
output = outputs[index_minScore] #Output file corresponding to the index
label = Labels[index_minScore]  #Labelled image corresponding to the index
label = label.reshape(image.shape) #Reshape image of labels to the original image's shape
# Read image
img = Image.open(output)
 
#Plot the graph for Cluster Quality Evaluation
plt.plot(Clusters,scores,'b--')
plt.plot(Clusters,scores,'bo')
plt.xlabel("Number of Clusters")
plt.ylabel("Davies Bouldin Score")
plt.title("Cluster Quality Evaluation")
plt.ylim(0.99*min(scores),1.01*max(scores))
plt.savefig("CQ_plot.png")
plt.close()
# Output Images
img.show()
print(f"Optimal Number of clusters = {C_opt} with Davies Bouldin score of {MinScore}")
img.close() #EDITED SW
#Plot the segmented image
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

ax0.imshow(image1, cmap='gray')
ax0.set_title('Original image')

ax1.imshow(label, cmap='gray')
ax1.set_title(f"Clustered Image for C = {C_opt} with Best Score of {MinScore:.4f}")

plt.savefig("Original_BestCluster.jpeg", dpi=300)
plt.close() #EDITED SW
