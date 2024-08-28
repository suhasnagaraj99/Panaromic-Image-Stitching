# Panaromic-Image-Stitching

## Project Description
This repository contains code for ENPM673 Assignment2 Question 2 - An image processing pipeline to stitch images to create a panoramic image.

![alt text](https://github.com/suhasnagaraj99/Panaromic-Image-Stitching/blob/main/Q2_Results/matches.jpg?raw=false)

![alt text](https://github.com/suhasnagaraj99/Panaromic-Image-Stitching/blob/main/Q2_Results/Pano.jpg?raw=false)

### Required Libraries
Before running the code, ensure that the following Python libraries are installed:

- `cv2`
- `numpy`
- `matplotlib`

You can install if they are not already installed:

```bash
sudo apt-get install python3-opencv python3-numpy python3-matplotlib
```

### Running the Code
Follow these steps to run the code:

1. Make sure the images `PA120272.JPG`,`PA120273.JPG`,`PA120274.JPG` and `PA120275.JPG` are pasted in the same directory as the `suhas99_project2_problem2.py` file.
2. Execute the `suhas99_project2_problem2.py`; run the following command in your terminal:

```bash
python3 suhas99_project2_problem2.py
```
3. The script matches the features between images and stitches them togeather to get a panoramic image as an output.
4. The script saves 2 images, `matches.jpg` and `Pano.jpg` as output.
