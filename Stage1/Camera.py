import cv2
import time

def capture_photos(camera_index, num_photos):
    print("Camera Open")
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not access the specified camera.")
        return

    # Set resolution to 640x480 (you can adjust this as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Adjust camera properties
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # Enable auto focus

    # Create a window for the live preview
    cv2.namedWindow("Live Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Preview", 640, 480)

    # Start the live preview loop
    photo_counter = 0
    while photo_counter < num_photos:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Failed to capture photo.")
            break

        # Display the live preview
        cv2.imshow("Live Preview", frame)

        # Check for key press to capture photo (space bar)
        key = cv2.waitKey(1)
        if key == ord(' '):
            # Save the captured frame to a file
            photo_counter += 1
            photo_filename = f'captured_photo_{photo_counter}.jpg'
            cv2.imwrite(photo_filename, frame)
            print(f"Photo {photo_counter} captured successfully: {photo_filename}")

#     # Release the webcam and close all windows
#     cv2.destroyAllWindows()
#
#     print("Live preview closed.")
#
# # Specify the index of the USB camera
# usb_camera_index = 1  # Change this to the index of your USB camera (it might be different)
#
# # Number of photos to capture
# num_photos = 10
#
# # Call the function to capture multiple photos
# capture_photos(usb_camera_index, num_photos)
