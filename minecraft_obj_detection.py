from ultralytics import YOLO
import cv2

class minecraft_obj_detection():

    def __init__(self, model_path='./yolov8_minecraft_weights.pt'):
        self.model = YOLO(model_path)
        self.window_name = "Detection window (press 'q' to close)"


    def rt_detection(self):
        
        import numpy as np
        import pyautogui

        # Define the coordinates for the screen region (x, y, width, height)
        x, y, width, height = int(1920 / 2), 65, int(1920 / 2), 1080
        
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)

        while True:
            # Capture the specified region of the screen
            screen = pyautogui.screenshot(region=(x, y, width, height))
            frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

            # Object detection
            results = self.model(frame)

            # Display the frame with predictions
            cv2.imshow(self.window_name, results[0].plot())

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def video_detection(self, video_path, output_path, display_detections=True):

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # Get video properties for output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create a VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


        while cap.isOpened():
            ret, frame = cap.read()        
            if not ret:
                break
            
            results = self.model(frame, verbose=False) # detect

            annotated_frame = results[0].plot()  # get the annotated frame

            out.write(annotated_frame) # write the annotated frame to the output

            if display_detections:
                cv2.imshow(self.window_name, annotated_frame) # display the annotated frame

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release the video capture and writer objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Output video saved as: {output_path}")