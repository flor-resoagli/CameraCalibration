import numpy as np
import cv2 as cv
import glob2 as glob
import pickle
import os
import yaml
import sys


def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()

def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()
    
    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])


# Define the chessboard size
chessboard_size = (8,8)

# Prepare the object points
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Create lists to store object points and image points from all images
object_points_list = []
imageRlist = []
imageLlist = []

imagesLeft = glob.glob('images/stereoLeft/*.png')
imagesRight = glob.glob('images/stereoRight/*.png')

# Load and process the images
for imgLeft, imgRight in zip(imagesLeft, imagesRight): 
    # image1 = cv2.imread('left_image{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread('right_image{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    cornersR = cv.goodFeaturesToTrack(grayR, maxCorners=64, qualityLevel=0.3, minDistance=7)
    # cornersR = np.int0(cornersR)

    cornersL = cv.goodFeaturesToTrack(grayL, maxCorners=64, qualityLevel=0.3, minDistance=7)
    # cornersL = np.int0(cornersL)

    # print(cornersR)



    # Load the precomputed corner points
    # Assuming you have numpy arrays "corners1" and "corners2" containing the corner points for each image pair
    # cornersL = np.load('cornersL_{}.npy'.format(i))
    # cornersR = np.load('cornersR_{}.npy'.format(i))

    # Add object points and image points to the lists

    # Ensure that cornersR and cornersL have the same number of corners
    if cornersR is not None and cornersL is not None and cornersR.shape[0] == cornersL.shape[0]:
        object_points_list.append(object_points)
        imageRlist.append(cornersR)
        imageLlist.append(cornersL)



    # object_points_list.append(object_points)
    # imageRlist.append(cornersR)
    # imageLlist.append(cornersL)

if len(object_points_list) > 0:


    # print(imageRlist)
    # Perform stereo calibration
    print(imgR.shape[-2:])
    ret, camera_matrix1, distortion_coefficients1, camera_matrix2, distortion_coefficients2, R, T, E, F = \
        cv.stereoCalibrate(object_points_list, imageRlist, imageLlist, None, None, None, None, imgR.shape[-2:])

    print("Stereo Calibration complete!")
    print("Camera Matrix 1:\n", camera_matrix1)
    print("Distortion Coefficients 1:\n", distortion_coefficients1)
    print("Camera Matrix 2:\n", camera_matrix2)
    print("Distortion Coefficients 2:\n", distortion_coefficients2)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    print("Essential Matrix:\n", E)
    print("Fundamental Matrix:\n", F)
else:
    print("No valid corner points found in any image pair.")





"""Step5. Save calibration data where camera0 defines the world space origin."""
#camera0 rotation and translation is identity matrix and zeros vector
R0 = np.eye(3, dtype=np.float32)
T0 = np.array([0., 0., 0.]).reshape((3, 1))

save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk
# R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
#check your calibration makes sense
# camera0_data = [cmtx0, dist0, R0, T0]
# camera1_data = [cmtx1, dist1, R1, T1]
# check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)



