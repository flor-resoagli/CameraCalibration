import numpy as np
import cv2 as cv
import glob2 as glob
import pickle



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8,8)
frameSize = (1280,960)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 21
objp = objp * size_of_chessboard_squares_mm
print(objp)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
# imgpointsL = [] # 2d points in image plane LEFT.
imgpointsR = [] # 2d points in image plane RIGHT.


# imagesLeft = glob.glob('images/stereoLeft/*.png')
imagesRight = glob.glob('images/stereoRight/*.png')





# for imgLeft, imgRight in zip(imagesLeft, imagesRight):
for imgRight in imagesRight:

    # imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)

    # grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # retL, thresholdL = cv.threshold(grayL, 200, 255, cv.THRESH_BINARY)
    retR, thresholdR = cv.threshold(grayR, 180, 255, cv.THRESH_BINARY)

    # cv.imshow('imgL', thresholdL)
    # cv.imshow('imgR', thresholdR)
    # cv.waitKey(1000)

    # Find the chess board corners
    # retL, cornersL = cv.findChessboardCorners(thresholdL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(thresholdR, chessboardSize, None)
    corners = cv.goodFeaturesToTrack(grayR, maxCorners=100, qualityLevel=0.3, minDistance=7)

    corners = np.int0(corners)

    print(corners)

    for corner in corners:
        
        x, y = corner.ravel()
        # print(str(x) + ',' + str(y))
        cv.circle(imgR, (x, y), 3, (0, 0, 255), -1)

    # Display the image
    cv.imshow('Corners', imgR)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # cv.imshow('imgR', figures)
    # cv.waitKey(1000)


    # print(cornersL)
    print(cornersR)


    print(str(retR) )

    # If found, add object points, image points (after refining them)
    # if retL and retR == True:
    if retR == True:

        objpoints.append(objp)
        # corners2L = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        # imgpoints.append(cornersL)
# 
        corners2R = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpoints.append(cornersR)

        # Draw and display the corners
        # cv.drawChessboardCorners(imgL, chessboardSize, corners2L, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, corners2R, retR)

        # cv.imshow('imgL', imgL)
        cv.imshow('imgR', imgR)
        cv.waitKey(1000)


cv.destroyAllWindows()




# ############## CALIBRATION #######################################################

# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
# pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
# pickle.dump(dist, open( "dist.pkl", "wb" ))


# ############## UNDISTORTION #####################################################

# img = cv.imread('cali5.png')
# h,  w = img.shape[:2]
# newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# # Undistort
# dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult1.png', dst)



# # Undistort with Remapping
# mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult2.png', dst)




# # Reprojection Error
# mean_error = 0

# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error

# print( "total error: {}".format(mean_error/len(objpoints)) )
