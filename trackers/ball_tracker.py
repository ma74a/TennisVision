from ultralytics import YOLO
import pickle
import os
import cv2

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)


    def interpolate_ball_positions(self, ball_detections):
        """Because the ball is not detected in all frames
        that cause a problem so we solve this problem using 
        interpolation
        """
        ball_positions = []
        for detection in ball_detections:
    
            if detection:
                bbox = list(detection.values())[0]
    
                x1, y1, x2, y2 = bbox
    
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
    
                w = x2 - x1
                h = y2 - y1
    
                ball_positions.append([cx, cy, w, h])
    
            else:
                ball_positions.append(None)
    
        # Interpolate missing positions
        for i in range(len(ball_positions)):
    
            if ball_positions[i] is not None:
                continue
    
            # Find previous detection
            prev = i - 1
    
            while prev >= 0 and ball_positions[prev] is None:
                prev -= 1
    
            # Find next detection
            next_ = i + 1
    
            while next_ < len(ball_positions) and ball_positions[next_] is None:
                next_ += 1
    
            if prev >= 0 and next_ < len(ball_positions):
    
                prev_pos = ball_positions[prev]
                next_pos = ball_positions[next_]
    
                alpha = (i - prev) / (next_ - prev)
    
                cx = prev_pos[0] + alpha * (next_pos[0] - prev_pos[0])
                cy = prev_pos[1] + alpha * (next_pos[1] - prev_pos[1])
    
                w = prev_pos[2] + alpha * (next_pos[2] - prev_pos[2])
                h = prev_pos[3] + alpha * (next_pos[3] - prev_pos[3])
    
                ball_positions[i] = [cx, cy, w, h]
    
        # Convert back to bounding boxes
        result = []
    
        for pos in ball_positions:
    
            if pos is None:
                result.append({})
                continue
    
            cx, cy, w, h = pos
    
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
    
            result.append({
                1: [x1, y1, x2, y2]
            })
    
        return result

    def detect_frames(self, frames, 
                          read_from_stub=False, 
                          stub_path=None
        ):
        """Detect video frames and store ball_track_id, bbox"""
        ball_detections = []
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_one_frame(frame=frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_one_frame(self, frame):
        """Detect One frame and get ball_track_id, and bboxs"""
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            results = box.xyxy.tolist()[0]

            ball_dict[1] = results

        return ball_dict


    def draw_boxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                # draw bounding boxes
                x1, y1, x2, y2 = bbox
                cv2.putText(img=frame, text=f"Ball ID: {track_id}", 
                            org=(int(bbox[0]),int(bbox[1] -10 )), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9, color=(225, 0, 0), thickness=2)
                
                cv2.rectangle(img=frame, pt1=(int(x1), int(y1)),pt2=(int(x2), int(y2)),
                                color=(255, 0, 0), thickness=2)

            output_video_frames.append(frame)

        return output_video_frames