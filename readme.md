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

### Lines detection

Table lines detection is based on the Probabilistic Hough Transform. Successive frames are transformed and detected lines are saved in the mask. When iteration is done, the mask contains the table lines and excess lines detected outside the table.

<p align="center">


  <img src="resources/readme/lines_table.gif?raw=true" alt="animated" />


</p>

### Choosing the right contours
On the finished mask, connected components are detected, and the component closest to the center of the image is selected.

<p align="center">

  <img src="resources/readme/lines_table_center.gif?raw=true" alt="animated" />

</p>

### Table line detection


</details>

</details>
