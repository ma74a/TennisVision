import cv2

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detect import CourtLineDetector
from mini_court import MiniCourt



def main():
    video_frames = read_video(video_path="/home/etman/etman/tennis_analysis/input_videos/input_video.mp4")

    player_tracker = PlayerTracker(model_path="models/yolov8m.pt")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    court_line_detector = CourtLineDetector(model_path="/home/etman/etman/tennis_analysis/models/keypoints_model.pth")
    # mini court object
    mini_court = MiniCourt(video_frames[0])

    players_detections = player_tracker.detect_frames(frames=video_frames,
                                                      read_from_stub=True,
                                                      stub_path="tracker_stubs/players_detections.pkl")
    

    ball_detections = ball_tracker.detect_frames(frames=video_frames,
                                                          read_from_stub=True,
                                                          stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # get keypoints
    keypoints = court_line_detector.predict_one_frame(image=video_frames[0])

    # just get the two players who play only
    players_detections = player_tracker.choose_and_filter_players(keypoints, players_detections)

    # # draw bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames=video_frames,
                                                    players_detections=players_detections)
    output_video_frames = ball_tracker.draw_boxes(video_frames=output_video_frames,
                                                        ball_detections=ball_detections)
    output_video_frames = court_line_detector.draw_points_on_video(output_video_frames, keypoints)

    # draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw frame number at the top left corner in the frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame ID: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # save the output video
    save_video(output_video_frames=output_video_frames, 
               output_video_path="output_videos/output_video.avi")



if __name__ == "__main__":
    main()