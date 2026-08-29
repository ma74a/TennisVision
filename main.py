import cv2
import pandas as pd
from copy import deepcopy

from trackers import PlayerTracker, BallTracker
from court_line_detect import CourtLineDetector
from mini_court import MiniCourt

from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants



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
    # # get the frames the ball is getting hit
    # frames = ball_tracker.get_ball_shot_frame(ball_detections=ball_detections)
    # print(f"frames: {frames}")

    # get keypoints
    keypoints = court_line_detector.predict_one_frame(image=video_frames[0])
    players_detections = player_tracker.choose_and_filter_players(keypoints, players_detections)
    # print(ball_detections)
    # return

    player_minicourt_detection, ball_minicourt_detection =  mini_court.convert_bounding_boxes_to_mini_court_coordinates(players_detections,
                                                                ball_detections,
                                                                keypoints)

    # print(player_minicourt_detection)


    # just get the two players who play only
    players_detections = player_tracker.choose_and_filter_players(keypoints, players_detections)

    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(players_detections,                                                                      ball_detections, keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']


    # # draw bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames=video_frames,
                                                    players_detections=players_detections)
    output_video_frames = ball_tracker.draw_boxes(video_frames=output_video_frames,
                                                        ball_detections=ball_detections)
    output_video_frames = court_line_detector.draw_points_on_video(output_video_frames, keypoints)

    # draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_minicourt_detection)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_minicourt_detection, color=[0,255, 255])
    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    # Draw frame number at the top left corner in the frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame ID: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # save the output video
    save_video(output_video_frames=output_video_frames, 
               output_video_path="output_videos/output_video.avi")



if __name__ == "__main__":
    main()