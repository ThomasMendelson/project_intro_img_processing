import cv2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory (directory of gui.py)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Get the parent directory (project directory)
sys.path.append(parent_dir)  # Add the parent directory to the Python path
from utils import colors, Masks, DEBUG, stock, long_side, IMG_HEIGHT, IMG_WIDTH, DEBUG2



def extract_straight_rect(image, coords):
    # Get the dimensions of the bounding box
    # width = int(rect[1][0])
    # height = int(rect[1][1])

    # Create a transformation matrix to straighten the rectangle

    # Reshape the coordinates array
    coords_reshaped = coords.reshape(4, 2)
    src_pts = coords_reshaped.astype("float32")
    dst_pts = np.array([[0, 399],
                        [0, 0],
                        [399, 0],
                        [399, 399]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the image to straighten the rectangle
    warped = cv2.warpPerspective(image, M, (399, 399))

    return warped


def find_close_up_image(image, color="Green"):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for the color mask
    lower_bound = np.array(Masks[color][0])
    upper_bound = np.array(Masks[color][1])
    mask_result = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    return extract_straight_rect(image, approx)


def are_similar_mse(img1, img2, threshold=0.95):
    # Compute Mean Squared Error (MSE) for each color channel
    mse_b = np.mean((img1[:, :, 0] - img2[:, :, 0]) ** 2)
    mse_g = np.mean((img1[:, :, 1] - img2[:, :, 1]) ** 2)
    mse_r = np.mean((img1[:, :, 2] - img2[:, :, 2]) ** 2)

    # Average MSE across color channels
    mse = (mse_b + mse_g + mse_r) / 3

    # Check if the SSIM is above the threshold
    # return ssim >= threshold
    return mse


def are_similar_using_orb(img1, img2, threshold=0.7):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score based on the number of good matches
    similarity = len(matches) / (len(kp1)+ 0.00000001)

    # Check if similarity is above the threshold
    return similarity


def are_similar_using_rgb_histogram(img1, img2, threshold=0.8):
    # Compute histograms for each color channel
    hist_b1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist_g1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    hist_r1 = cv2.calcHist([img1], [2], None, [256], [0, 256])

    hist_b2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_g2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    hist_r2 = cv2.calcHist([img2], [2], None, [256], [0, 256])

    # Concatenate histograms for each color channel
    hist1 = np.concatenate((hist_b1, hist_g1, hist_r1))
    hist2 = np.concatenate((hist_b2, hist_g2, hist_r2))

    # Normalize histograms
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    # Compute histogram intersection
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    # Check if intersection is above the threshold
    return intersection  # >= threshold


def remove_bg(img, color="Green"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for the color mask
    lower_bound = np.array(Masks[color][0])
    upper_bound = np.array(Masks[color][1])
    mask_result = cv2.inRange(hsv, lower_bound, upper_bound)
    # Invert the mask
    mask_result = cv2.bitwise_not(mask_result)
    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask_result)
    return result


def calc_score(img, building):
    max_score = 50
    success = 0
    max_score_txt = ""
    txt = ""
    building_path = f"../Data Base/Library/shapes/{building.name}"
    try:
        img = find_close_up_image(img)
        img = remove_bg(img)
        img = centerImage(img)

        for filename in os.listdir(building_path):
            success = 0
            img2 = cv2.imread(building_path + "/" + str(filename))
            img2 = find_close_up_image(img2)
            img2 = remove_bg(img2)
            img2 = centerImage(img2)
            if DEBUG:
                # Create two separate windows and display the images
                cv2.imshow('live 1', img)
                cv2.imshow('ref 2', img2)

                # Wait for a key press and then close all windows
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            mse = are_similar_mse(img, img2)
            orb = are_similar_using_orb(img, img2)
            his = are_similar_using_rgb_histogram(img, img2)
            score = (100 - mse) * 0.33 + his * 33 + 33 * orb
            if building.name == "RedCross" or building.name == "Pyramid":
                if mse <= 3.5:
                    success += 1
                if his > 0.98:
                    success += 1
                if orb > 0.48:
                    success += 1
            elif building.name == "Balloon" or building.name == "Invader" or building.name == "Mushroom" or building.name == "USAFlag" or building.name == "Car":
                if mse <= 9:
                    success += 1
                if his > 0.97:
                    success += 1
                if orb > 0.42:
                    success += 1
            elif building.name == "House":
                if mse <= 23:
                    success += 1
                if his > 0.97:
                    success += 1
                if orb > 0.42:
                    success += 1
            elif building.name == "Tammy":
                if mse <= 23:
                    success += 1
                if his > 0.92:
                    success += 1
                if orb > 0.5:
                    success += 1
            elif building.name == "Smily":
                if mse <= 13:
                    success += 1
                if his > 0.97:
                    success += 1
                if orb > 0.42:
                    success += 1
            else:
                if mse <= 5.5:
                    success += 1
                if his > 0.98:
                    success += 1
                if orb > 0.42:
                    success += 1
            txt = f"mse: {mse:.5f} his: {his:.5f} orb: {orb:.5f}"
            # txt = f"mse: {mse:.5f} his: {his:.5f} orb: {orb:.5f}  Score: {score:.5f}"
            if score > max_score:
                max_score = score
                max_score_txt = txt
                if success >= 2:
                    max_score_txt = txt + "    \n   You did it!!!    Good Job"
        return max_score_txt, str(max_score)
    except:
        return "Could not recognize", 0



def centerImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to isolate the object
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (assuming it's the object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the contour
    M = cv2.moments(largest_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    # Calculate the offset to center the object in the image
    image_height, image_width = image.shape[:2]
    offset_x = (image_width // 2) - centroid_x
    offset_y = (image_height // 2) - centroid_y

    # Translate the object by the offset
    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image_width, image_height))

    return translated_image


def t():
    img1 = cv2.imread("../Data Base/Library/shapes/IR/IR_2.jpg")
    img1 = find_close_up_image(img1)

    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.show()

    for filename2 in os.listdir('../Data Base/Library/shapes/test'):
        img2 = cv2.imread("../Data Base/Library/shapes/test/" + str(filename2))
        try:
            img2 = find_close_up_image(img2)
        except Exception:
            print("Error in ", filename2)
            continue
        # remove back ground
        img1 = remove_bg(img1)
        img2 = remove_bg(img2)
        # img2 = align_images(img2, img1)
        mse = are_similar_mse(img1, img2)
        orb = are_similar_using_orb(img1, img2)
        his = are_similar_using_rgb_histogram(img1, img2)
        match = (mse < 40 and his > 0.91 and orb > 0.1)
        if not match:
            continue
        else:
            # Create a figure and axes for the subplots
            fig, axes = plt.subplots(1, 2)
            # Display the first image on the first subplot
            axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

            # Display the second image on the second subplot
            axes[0].set_title(str(match))
            axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axes[1].set_title(str(mse) + "\nORB:" + str(orb)
                              + "\nHIST:" + str(his))
            plt.show()
        # plt.waitforbuttonpress()


def align_images(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix using RANSAC with higher reprojThreshold
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 99.0)

    # Warp img1 onto img2
    aligned_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    return aligned_img



if __name__ == "__main__":
    t()