import torch
from torchvision import models, transforms
import cv2


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_one_frame(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor).squeeze(0)

        keypoints = output.cpu().numpy()
        original_h, original_w = img.shape[:2]

        # print(original_h, original_w)
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_points_on_frame(self, frame, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        return frame

    def draw_points_on_video(self, frames, keypoints):
        output_video_frames = []
        for frame in frames:
            output_video_frames.append(self.draw_points_on_frame(frame, keypoints))

        return output_video_frames


