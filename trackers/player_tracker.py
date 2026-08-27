from ultralytics import YOLO
import cv2
from typing import List, Dict
import pickle
import os

from utils import get_box_center, measure_distance, get_foot_position

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)

    def choose_and_filter_players(self, court_keypoints, players_detections):
        """filter players to delete bbox around people except who play only"""
        player_dict_for_frame = players_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_dict_for_frame)

        filtered_player_detections = []
        for player_dict in players_detections:
            filtered_players = {track_id: bbox 
                                for track_id, bbox in player_dict.items() 
                                if track_id in chosen_players}
            filtered_player_detections.append(filtered_players)

        # print(filtered_player_detections)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        """Choose the two detected people whose centers are closest to the tennis court keypoints."""
        chosen_players = []
        for track_id, bbox in player_dict.items():
            player_center = get_box_center(bbox)

            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance

            chosen_players.append((track_id, min_distance))

        # sort the distances in ascending order
        chosen_players.sort(key=lambda x: x[1])
        # get the track_id of min_distance_players the first 2
        min_distance_players = [chosen_players[0][0], chosen_players[1][0]]
        print(min_distance_players)

        return min_distance_players


    def detect_frames(self, frames, 
                      read_from_stub=False, 
                      stub_path=None
    ) -> List[Dict[int, List[float]]]:
        """Detect video frames and store player_track_id, bbox"""
        players_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                players_detections = pickle.load(f)
            return players_detections
        
        for frame in frames:
            player_dict = self.detect_one_frame(frame=frame)
            players_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(players_detections, f)

        return players_detections

    def detect_one_frame(self, frame) -> Dict[int, List[float]]:
        """Detect One frame and get player_track_id, and bboxs"""
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {} # player_id -> bounding box
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_boxes(self, video_frames, players_detections):
        """Draw bboxes around each player"""
        output_video_frames = []
        for frame, player_dict in zip(video_frames, players_detections):
            for track_id, bbox in player_dict.items():
                # draw bounding boxes
                x1, y1, x2, y2 = bbox
                cv2.putText(img=frame, text=f"Player ID: {track_id}", 
                            org=(int(bbox[0]),int(bbox[1] -10 )), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9, color=(0, 0, 255), thickness=2)
                
                cv2.rectangle(img=frame, pt1=(int(x1), int(y1)),pt2=(int(x2), int(y2)),
                              color=(0, 0, 255), thickness=2)

            output_video_frames.append(frame)

        return output_video_frames