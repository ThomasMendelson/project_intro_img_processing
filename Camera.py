import cv2


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
            # if photo_counter < 10:
            #     reduction = 0
            #     color = "Green"
            # elif photo_counter < 20:
            #     color = "Yellow"
            #     reduction = 10
            # elif photo_counter < 30:
            #     color = "Black"
            #     reduction = 20
            # elif photo_counter < 40:
            #     color = "Blue"
            #     reduction = 30
            # elif photo_counter < 50:
            #     color = "Red"
            #     reduction = 40
            # elif photo_counter < 60:
            #     color = "Orange"
            #     reduction = 50
            # elif photo_counter < 70:
            #     color = "White"
            #     reduction = 60
            # num = photo_counter-reduction
            photo_filename = f'O_{photo_counter}.jpg'
            cv2.imwrite(photo_filename, frame)
            print(f"Photo {photo_counter} captured successfully: {photo_filename}")
            photo_counter += 1

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

def capture_video(camera_index, video_duration):
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
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable auto focus

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec to be used for video writing
    out = cv2.VideoWriter('video0.avi', fourcc, 20.0, (640, 480))  # Output file name, codec, frame rate, frame size

    # Start capturing and writing frames for the specified duration
    start_time = cv2.getTickCount()  # Get the initial time
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Failed to capture frame.")
            break

        out.write(frame)  # Write the frame to the output video

        cv2.imshow('frame', frame)  # Display the frame

        # Check for time elapsed
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - start_time) / cv2.getTickFrequency()
        if time_elapsed >= video_duration:
            break

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video capture completed.")