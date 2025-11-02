from pathlib import Path
from src.config import TrainConfig
from src.inference import predict_age_rating, predict_age_rating_with_scores

cfg = TrainConfig()
video = Path("data/videos/-53DvfE42gE.mp4")
checkpoint = Path("runs/slowfast/best.pt")

label = predict_age_rating(video, checkpoint, cfg)
scores = predict_age_rating_with_scores(video, checkpoint, cfg)
print(label)
print(scores)