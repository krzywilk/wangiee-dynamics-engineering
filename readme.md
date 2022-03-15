# Table tennis entities detection

<p align="center">
  <img src="resources/readme/base_detections.gif?raw=true" alt="animated" />
</p>

## Introduction
The algorithm is designed to detect the elements of table tennis gameplay.
The test dataset contains videos recorded from a static position. The camera is always behind the back of one of the players.


Table tennis entities detection is based only on the classic computer vision and clustering algorithms.

## Scripts execution 
<details>
<summary> <b>Installation</b> </summary>
Due to the use of only image operations and unsupervised clustering algorithms, the GPU is not required. 
To prepare the environment, just install the libraries from requirements.txt.
</details>
<details>
<summary> <b>Running</b> </summary>
Temporarily there is no specific script configuration. An example usage is in the main.py file.
</details>

## Output
The result of the algorithm are points representing the coordinates of corners of the table, and the players bounding boxes.



<div align="center">

|                             Top player                             |                              Bottom player                               |
|:------------------------------------------------------------------:|:------------------------------------------------------------------------:|
| ![Alt text](resources/readme/player_top.gif?raw=true "top player") | ![Alt text](resources/readme/player_bottom.gif?raw=true "bottom player") |


![Alt text](resources/readme/table.gif?raw=true "table")

</div>

## Algorithm steps details

<details>
<summary><b> Table corners detection </b></summary>

<details>
<summary> Table contours detection </summary>

### Table lines detection

#### Lines detection

Table lines detection is based on the Probabilistic Hough Transform. Successive frames are transformed and detected lines are saved in the mask. When iteration is done, the mask contains the table lines and excess lines detected outside the table.

<p align="center">
  <img src="resources/readme/lines_table.gif?raw=true" alt="animated" />
</p>

#### Choosing the right contours
On the finished mask, connected components are detected, and the component closest to the center of the image is selected.

<p align="center">
  <img src="resources/readme/lines_table_center.gif?raw=true" alt="animated" />
</p>


</details>

<details>
<summary> Table mask corner detections</summary>

### Lines parameters detection
<p align="center">
  <img src="resources/readme/lines_mask.gif?raw=true" alt="animated" />
</p>

#### General line detection
Mask lines detection is based on Hough Line Transform.

<p align="center">
  <img src="resources/readme/hough_line_transform.jpg?raw=true" alt="animated" />
</p>

#### Lines filtering
The algorithm filters redundant lines which rho and theta values ​​are close to each other.

<p align="center">
  <img src="resources/readme/hough_line_transform_filtered.jpg?raw=true" alt="animated" />
</p>

#### Lines clustering
The lines are clustered due to the angle of inclination. Clustering is done by the DBSCAN algorithm. Outlier lines are removed from lines list.

<p align="center">
  <img src="resources/readme/hough_line_transform_filtered_clustered.jpg?raw=true" alt="animated" />
</p>


### Lines intersection on a table

<p align="center">
  <img src="resources/readme/intersections.gif?raw=true" alt="animated" />
</p>

#### Intersections
All points of intersection between the horizontal and vertical lines are calculated basis on theta i rho of lines.

<p align="center">
  <img src="resources/readme/intersections.jpg?raw=true" alt="animated" />
</p>


#### Intersections clustering
The Intersections are clustered due to the position on the Cartesian plane. Clustering is done by the DBSCAN algorithm.

<p align="center">
  <img src="resources/readme/intersections_clusters.jpg?raw=true" alt="animated" />
</p>


#### Intersections clusters centroids
Intersection cluster centroids are calculated as the average of all existing points in the cluster.

<p align="center">
  <img src="resources/readme/intersections_centroids.jpg?raw=true" alt="animated" />
</p>

</details>



</details>
