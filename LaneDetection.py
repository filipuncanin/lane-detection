import cv2 as cv
import numpy as np
import os
import glob

source = "test_videos/project_video01.mp4"
frame_width = 1280
frame_height = 720

def camera_calibration():
    if not os.path.exists('camera_calibration.npz'):
        # Parameters for the chessboard pattern
        chessboard_size = (9, 6)    # Number of internal corners (rows, cols)
        square_size = 1.0           # Size of a square (arbitrary units, e.g., meters or millimeters)

        # Prepare object points, e.g., (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # Arrays to store object points and image points
        object_points = []  # 3D points of the chessboard in real world space
        image_points = []   # 2D points in image plane

        # Get list of calibration images
        images = glob.glob('camera_cal/*.jpg')

        for idx, filename in enumerate(images):
            # Read image
            image = cv.imread(filename)
            if image.shape[0] != frame_height or image.shape[1] != frame_width:
                image = cv.resize(image, (frame_width, frame_height))
                
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)    # Locate the internal corners of the chessboard

            if ret:
                object_points.append(objp)
                image_points.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(image, chessboard_size, corners, ret)
                cv.imshow('Calibration', image)
                cv.waitKey(1000)

        cv.destroyAllWindows()

        # Perform camera calibration
        ret, camera_matrix, distortion_coeffs, _, _ = cv.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None
        )

        if ret:
            print("Camera Calibration Successful!")
            np.savez('camera_calibration.npz', camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)
            return camera_matrix, distortion_coeffs

        else:
            print("Camera Calibration Failed!")
            return -1
        
    else:
        data = np.load('camera_calibration.npz')
        camera_matrix, distortion_coeffs, rvecs, tvecs = data['camera_matrix'], data['distortion_coeffs'], data.get('rvecs', None), data.get('tvecs', None)
        return camera_matrix, distortion_coeffs

def distortion_correction(frame, camera_matrix, distortion_coeffs):
    return cv.undistort(frame, camera_matrix, distortion_coeffs, None, camera_matrix)

def threshold_image(frame):
    hsv_transformed_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    yellow_lower = np.array([18, 94, 140])
    yellow_upper = np.array([30, 255, 255])

    white_mask = cv.inRange(hsv_transformed_frame, white_lower, white_upper)
    yellow_mask = cv.inRange(hsv_transformed_frame, yellow_lower, yellow_upper)

    return cv.bitwise_or(white_mask, yellow_mask)

def perspective_transformation(frame):
    img = np.copy(frame)

    # Choosing points for perspective transformation
    top_left = (frame.shape[1]//50*22,frame.shape[0]//8*5)
    top_right = (frame.shape[1]//50*29,frame.shape[0]//8*5)
    bottom_left = (0,frame.shape[0])
    bottom_right = (frame.shape[1],frame.shape[0])

    ## Aplying perspective transformation
    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
    
    # Matrix to warp the image for birdseye window
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    matrixInv = cv.getPerspectiveTransform(pts2, pts1) 

    return cv.warpPerspective(img, matrix, (640,480)), matrixInv

def sliding_windows(frame):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(frame[frame.shape[0]//2:, :], axis=0) 

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((frame, frame, frame)) * 255

    # Find the peaks of the histogram to locate the starting points for the left and right lines
    midpoint = np.int_(histogram.shape[0] // 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set hyperparameters for sliding windows
    nwindows = 12          # Number of sliding windows

    # Set height of windows
    window_height = np.int_(frame.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    left_current = left_base
    right_current = right_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y
        win_y_low = frame.shape[0] - (window + 1) * window_height
        win_y_high = frame.shape[0] - window * window_height
        win_xleft_low = left_current - 100
        win_xleft_high = left_current + 100
        win_xright_low = right_current - 100
        win_xright_high = right_current + 100

        # Draw the windows on the visualization image
        cv.rectangle(out_img, (win_xleft_low, win_y_low),
                     (win_xleft_high, win_y_high), (255, 255, 255), 2)
        cv.rectangle(out_img, (win_xright_low, win_y_low),
                     (win_xright_high, win_y_high), (255, 255, 255), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > 50:
            left_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > 50:
            right_current = np.int_(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Draw the lane onto the warped blank image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 255, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]

    for i, y in enumerate(ploty.astype(int)):
        if y < frame.shape[0]:
            cv.circle(out_img, (int(left_fitx[i]), int(y)), 2, (0, 0, 255), -1)
            cv.circle(out_img, (int(right_fitx[i]), int(y)), 2, (0, 0, 255), -1)

    return out_img, left_fit, right_fit

def measure_curvature_and_position(binary_warped, left_fit, right_fit):
    # Define conversions in x and y from pixels to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)  # Evaluate curvature at the bottom of the image

    # Refit the polynomials to world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fit[0] * ploty**2 * xm_per_pix + left_fit[1] * ploty * xm_per_pix + left_fit[2] * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fit[0] * ploty**2 * xm_per_pix + right_fit[1] * ploty * xm_per_pix + right_fit[2] * xm_per_pix, 2)

    # Calculate radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])

    # Calculate the vehicle's position with respect to the center
    car_center = binary_warped.shape[1] / 2  # Image center
    left_lane_bottom = left_fit[0] * (binary_warped.shape[0] - 1)**2 + left_fit[1] * (binary_warped.shape[0] - 1) + left_fit[2]
    right_lane_bottom = right_fit[0] * (binary_warped.shape[0] - 1)**2 + right_fit[1] * (binary_warped.shape[0] - 1) + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom) / 2

    vehicle_position = (car_center - lane_center) * xm_per_pix

    return left_curverad, right_curverad, vehicle_position

def draw(original_img, binary_warped, left_fit, right_fit, Minv, left_curverad, right_curverad, vehicle_position):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
    newwarp = cv.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv.addWeighted(original_img, 1, newwarp, 0.5, 0)

    curvature_text = f"Curvature: Left = {left_curverad:.2f} m, Right = {right_curverad:.2f} m"
    position_text = f"Vehicle Position: {abs(vehicle_position):.2f} m {'left' if vehicle_position < 0 else 'right'} of center"

    # Overlay text on the frame
    cv.putText(result, curvature_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(result, position_text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    return result

if __name__ == '__main__':

    # 1. Camera calibration
    camera_matrix, distortion_coeffs = camera_calibration()

    capture = cv.VideoCapture(source)
    fps = capture.get(cv.CAP_PROP_FPS)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
            frame = cv.resize(frame, (frame_width, frame_height))

        # 2. Apply a distortion correction to raw images.
        undistorted_frame = distortion_correction(frame, camera_matrix, distortion_coeffs)

        # 3. Use color transforms, gradients, etc., to create a thresholded binary image.
        thresholded_frame = threshold_image(undistorted_frame)

        # 4. Apply a perspective transform to rectify binary image (“birds-eye view”).
        perspective_frame, mInv = perspective_transformation(thresholded_frame)

        # 5. Sliding Windows
        sliding_window_frame, left_fit, right_fit = sliding_windows(perspective_frame)

        # Determine the curvature of the lane and vehicle position with respect to center.
        left_curverad, right_curverad, vehicle_position = measure_curvature_and_position(perspective_frame, left_fit, right_fit)

        # 6. Draw
        final_frame = draw(undistorted_frame, perspective_frame, left_fit, right_fit, mInv, left_curverad, right_curverad, vehicle_position)

        # cv.imshow('Original Frame', frame)
        # cv.imshow('Undistorted Frame', undistorted_frame)
        # cv.imshow('Thresholded Frame', thresholded_frame)
        # cv.imshow('Perspective Transformation Frame', perspective_frame)
        # cv.imshow('Sliding Windows Frame', sliding_window_frame)
        cv.imshow('Final Frame', final_frame)

        if cv.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()