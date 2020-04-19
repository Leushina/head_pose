# Head Pose Estimation

This is a script that will estimate head position from video or live stream. Landmarks for estimation are found with pretrained model
[from here](https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat). 
World coordinates are taken [from here](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/#code), 
as well as camera parameters approximation. 



### 1. To run this script 
Before running, make sure that you have "shape_predictor_68_face_landmarks.dat" file with pretrained model for landmarks in the same folder as the script.


To process video from your webcam (to exit, press 'q'): 
```
head_pose.py --cam 0 
```

or to process file: 

```
head_pose.py --video filename 
```
