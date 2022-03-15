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
Top player|                                Bottom player                                
:-------------------------:|:------------------------------------------------------------------------:
![Alt text](resources/readme/player_top.gif?raw=true "top player") | ![Alt text](resources/readme/player_bottom.gif?raw=true "bottom player")

