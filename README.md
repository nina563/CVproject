# Camera calibration using two vanishing points
In this project, I reproduced the [approach](https://www.researchgate.net/publication/226987370_Using_Vanishing_Points_for_Camera_Calibration_and_Coarse_3D_Reconstruction_from_A_Single_Image), 
that utilizes two vanishing points produced by the planar calibration object. The goal is to determine the extrinsic parameters for six cameras, placed around the perimeter of a room.  


### Tech used
- [Python3](https://www.python.org)
- [NumPy](https://numpy.org)
- [opencv-python](https://opencv.org)

### Project Structure
1. camera_calibration - contains code for computing camera positions and intrinsic matrices.
2. callibration_patterns.py -contains parameters of the employed planAR checkerboard pattern.
3. load.py- contains code for loading images, used for running the project.
5. scene_calibration.py - contains code for retrieving extrinsic matrices of all the cameras

### Running the project
Run scene_calibration.py using the below to retrieve extrinsic matrices from all 6 test images.   
```
python scene_calibration.py
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

