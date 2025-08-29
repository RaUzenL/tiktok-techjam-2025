import kagglehub
import shutil
import os

# Download dataset
dataset_path = kagglehub.dataset_download("denizbilginn/google-maps-restaurant-reviews")
print("Downloaded dataset path:", dataset_path)

# Copy the main CSV to your repo's data/ folder
os.makedirs("data", exist_ok=True)
shutil.copy(f"{dataset_path}/reviews.csv", "data/reviews.csv")
shutil.copy(f"{dataset_path}/sepetcioglu_restaurant.csv", "data/sepetcioglu_restaurant.csv")
