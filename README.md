These are the files/code of my camera calib using OpenCV-Python. These code files are not so much organized as I did not find spare time to clean the code or write a good documentation/tutorial.

# lib

- opencv 4.5.5
- Pyqt5
- yaml
- Numpy

# tab Calib Real time camera calibration tool

Calib matrixes, dist of camera from CHESSBOARD

- Start Button : enable calib
- Detect Corners: enable Corners
- IGNORE: IGNORE image
- CONFIRM: use image 
- Done : begin calib with num images take from camera
- 

![1](https://user-images.githubusercontent.com/8399429/151115641-098ccaf0-71aa-48af-b2de-05516ff4b569.png)

# tab Calib World 

- Calib real world (Just XY ) from  pixels position
- Use 7 points
- Step1: calculator position of points from realworld 
- Step2: calculator position of points from pixel
- detect points to help detecting a object pixel,which will help to step2
- Button calib: calib function


https://user-images.githubusercontent.com/8399429/151115533-a7619e86-1f6d-4f67-b3ca-3fdaae69a345.mp4



https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/

 
# Tab  Run Test

- Run demo to detect objects with real-world-coordinates

https://user-images.githubusercontent.com/8399429/151115495-970a31bd-dc1f-4602-acd0-a1d30493505b.mp4
