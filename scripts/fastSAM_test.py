#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import time
from ultralytics import FastSAM
### Always import torch and ultralytics before any ROS-related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FastSAMSegmenter:
    def __init__(self):
        rospy.init_node('fastsam_segment_node', anonymous=True)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.segmented_frame = None
        self.frame_count = 0
        self.detection_interval = 3

        # Load FastSAM model
        rospy.loginfo("Loading FastSAM model...")
        self.model = FastSAM('FastSAM-s.pt')  # Or use 'FastSAM-s.pt' for smaller model

        if torch.cuda.is_available():
            rospy.loginfo("CUDA is available, using GPU")
            self.model.to('cuda')
        else:
            rospy.logwarn("CUDA not available, using CPU (may be slow)")

        # Subscribe to ROS image topic
        self.image_sub = rospy.Subscriber(
            '/rs_grippercam/color/image_raw', Image, self.image_callback, queue_size=1)

        rospy.loginfo("FastSAM segmenter initialized")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    # def run_segmentation(self):
    #     if self.latest_frame is None:
    #         return None

    #     frame = self.latest_frame.copy()
    #     start_time = time.time()

    #     # Run segmentation
    #     results = self.model(frame, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True)

    #     # Plot the first result (there should only be one for a single frame)
    #     # annotated_image = results[0].plot(mask_alpha=0.4)
    #     annotated_image = results[0].plot()

    #     fps = 1.0 / (time.time() - start_time)
    #     # cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 30),
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #     print(f"fps: {fps:.1f}")

    #     return annotated_image
            
    def run_segmentation(self):
        if self.latest_frame is None:
            return None

        frame = self.latest_frame.copy()
        start_time = time.time()

        # Run FastSAM inference
        results = self.model(
            frame, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True
        )

        # Get masks and filter by area
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data  # Shape: [N, H, W]
            min_area = 200   # minimum pixel count
            max_area = 100000 # maximum pixel count

            filtered_masks = []
            for mask in masks:
                area = mask.sum().item()
                if min_area <= area <= max_area:
                    filtered_masks.append(mask.cpu().numpy().astype(np.uint8))

            # Overlay filtered masks on the frame
            color_mask = np.zeros_like(frame)
            for mask in filtered_masks:
                color_mask[mask == 1] = (0, 255, 0)  # Green

            annotated_image = cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0)
        else:
            rospy.logwarn("No masks found in FastSAM results")
            annotated_image = frame

        fps = 1.0 / (time.time() - start_time)
        print(f"fps: {fps:.1f}")

        return annotated_image

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.latest_frame is not None:
                if self.frame_count % self.detection_interval == 0:
                    self.segmented_frame = self.run_segmentation()

                if self.segmented_frame is not None:
                    cv2.imshow("FastSAM Segmentation", self.segmented_frame)
                else:
                    cv2.imshow("Camera Feed", self.latest_frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break

                self.frame_count += 1

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        segmenter = FastSAMSegmenter()
        segmenter.run()
    except rospy.ROSInterruptException:
        pass














# ###############################################
# # USING WEBCAM
# ###############################################

# #!/usr/bin/env python3

# import cv2
# import torch
# import time
# from ultralytics import FastSAM

# class FastSAMWebcamDemo:
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             raise RuntimeError("Could not open webcam")

#         # Load FastSAM model
#         print("[INFO] Loading FastSAM model...")
#         self.model = FastSAM('FastSAM-x.pt')  # Or 'FastSAM-s.pt' for smaller version

#         # Move to GPU if available
#         if torch.cuda.is_available():
#             print("[INFO] Using GPU")
#             self.model.to('cuda')
#         else:
#             print("[WARN] CUDA not available, using CPU (slower)")

#     def run(self):
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("[ERROR] Failed to grab frame")
#                 break

#             start_time = time.time()

#             # Run segmentation
#             resized_frame = cv2.resize(frame, (512, 512))
#             results = self.model(resized_frame, device='cuda' if torch.cuda.is_available() else 'cpu')

#             # Overlay the masks
#             annotated_image = self.model.plot(
#                 results=results,
#                 image=resized_frame,
#                 mask_alpha=0.4
#             )

#             fps = 1.0 / (time.time() - start_time)
#             cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             cv2.imshow("FastSAM Webcam", annotated_image)

#             key = cv2.waitKey(1)
#             if key == 27:  # ESC to exit
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     demo = FastSAMWebcamDemo()
#     demo.run()
