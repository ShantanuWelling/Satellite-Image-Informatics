Topic 3. Implement FCM clustering algorithm and evaluate the cluster quality for different values of ‘C’. (Minimum and maximum values of C to be specified by user) 

Team Members: Shantanu Welling (210010076), Harshit Agarwal (210020054), Arijit Saha (210050017) <br>
​
Requirements: Python libraries- numpy, matplotlib, scikit-learn, scikit-image, scipy, PIL (Pillow), sys

Run executable python script named fcm.py from Linux terminal as:
python3 fcm.py {input_image_name} {min_clusters} {max_clusters}

The input image name must be specified as a path relative to the directory in which the script fcm.py is placed in.

Prints optimal number of clusters along with the corresponding Davies Bouldin Score as "Optimal Number of clusters = 6 with Davies Bouldin score of 0.49954825815125287" on the terminal. 

Also saves (in the same directory as that of the script) the clustering images for each value of C as "output_C{i}.jpeg" where i is the number of clusters used for clustering the image. 

It also saves the original image and the optimal clustered image (side by side in the same image) in the file named "Original_BestCluster.jpeg" in the same directory as that of the script.

Also saves the corresponding graph for the input image which plots Number of clusters vs Davies Bouldin score values for the corresponding clustering in a file named "CQ_plot.png" in the same directory as that of the script.

Folder named "results" in this zip file contains subfolders named "img{i}" where i corresponds to the input file named "img{i}.png"/"img{i}.jpeg" that was used. There are 15 images inside each subfolder.
The 15 images in subfolder "img{i}" correspond to: <br>
13 clustering images obtained (output_C{i}.jpeg where i ranges from 3 to 15- both inclusive) <br>
1 Graph plot (CQ_plot.png) <br>
1 Original vs Best Clustered image (Original_BestCluster.jpeg) <br>

These 15 images of subfolder "img{i}" are obtained after executing the following command on terminal:<br>
"python3 fcm.py img{i}.png 3 15" or "python3 fcm.py img{i}.jpeg 3 15" depending on whether the img is in jpeg or png data format.

Each input image has range of cluster values between 3 and 15 (both inclusive) as the user input parameters.

There are 8 such subfolders inside "results" folder corresponding to the results obtained from the 8 input images in the zip file's main (parent) directory (img{i} where i ranges from 1 to 8-both inclusive) (having .png/.jpeg extension)

There is a res.txt file in the main (parent) directory which has the Optimal Cluster number along with the corresponding Davies Bouldin Score for each of the 8 input images used. (i.e. the output printed on terminal for each input image).
