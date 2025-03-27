# %% [markdown]
# # Train YOLOv8 with labeled data from Roboflow

# %%
# 1. YOLOv8 Training for Israeli Police Vehicle Detection

# Set locale to UTF-8 to avoid NotImplementedError
import os

!pip install ultralytics

# %%
# 2. Import libraries
from google.colab import drive
from ultralytics import YOLO
from tabulate import tabulate
from IPython.display import display
import os
import shutil
import zipfile
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# %% [markdown]
# ## Roboflow dataset extraction
# - searching in google drive
# - pulling the zip file and extract it
# - the roboflow file includes: train, valid, test data sets with labels annotations and data.yams file.
# - data.yams file is a configuration file used by YOLO models to understand the dataset structure and classes.

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import zipfile
import glob

# Find zip file in Google Drive
drive_folder = '/content/drive/MyDrive'

zip_files = []
for root, dirs, files in os.walk(drive_folder):
    for file in files:
        if 'police' in file.lower() and file.endswith('.zip'):
            zip_files.append(os.path.join(root, file))
            print(f"Found potential dataset: {os.path.join(root, file)}")

if not zip_files:
    print("No dataset zip file found. Please upload it to your Google Drive.")
    # List some files to help locate it
    print("\nFiles in MyDrive root:")
    for file in os.listdir(drive_folder):
        print(f"  {file}")
else:
    # If zip file found, extract it
    if len(zip_files) > 1:
        print(f"\nFound {len(zip_files)} potential zip files. Using the first one: {zip_files[0]}")

    zip_path = zip_files[0]
    print(f"\nExtracting {zip_path} to /content...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content')

    print("Extraction completed!")

    # Check the structure of extracted files
    print("\nChecking extracted files structure:")
    subdirs = [d for d in os.listdir('/content') if os.path.isdir(os.path.join('/content', d))]
    print(f"Directories in /content: {subdirs}")

    # Check for train/valid/test folders
    for folder in ['train', 'valid', 'test']:
        if os.path.exists(f'/content/{folder}'):
            images_dir = f'/content/{folder}/images'
            labels_dir = f'/content/{folder}/labels'

            if os.path.exists(images_dir):
                print(f"{folder} images: {len(os.listdir(images_dir))}")
            else:
                print(f"{folder} images directory not found")

            if os.path.exists(labels_dir):
                print(f"{folder} labels: {len(os.listdir(labels_dir))}")
            else:
                print(f"{folder} labels directory not found")
        else:
            print(f"{folder} directory not found")

    # Check for data.yaml
    if os.path.exists('/content/data.yaml'):
        print("\ndata.yaml file found!")
        with open('/content/data.yaml', 'r') as f:
            print("Content:")
            print(f.read())
    else:
        print("\ndata.yaml file not found")

# %% [markdown]
# ## Creating runtime local datasets paths for easy and fast reach.

# %%
# 3. Mount Google Drive (for saving results)
drive.mount('/content/drive')

# 4. Define dataset paths based on the actual structure
DATASET_DIR = '/content'  # Your files are directly in the content folder
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALID_DIR = os.path.join(DATASET_DIR, 'valid')  # Note: folder is named 'valid' not 'validation'
TEST_DIR = os.path.join(DATASET_DIR, 'test')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

# 5. Verify the dataset structure
def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

print("Dataset Structure:")
print(f"Training images: {count_files(os.path.join(TRAIN_DIR, 'images'))}")
print(f"Training labels: {count_files(os.path.join(TRAIN_DIR, 'labels'))}")
print(f"Validation images: {count_files(os.path.join(VALID_DIR, 'images'))}")
print(f"Validation labels: {count_files(os.path.join(VALID_DIR, 'labels'))}")
print(f"Test images: {count_files(os.path.join(TEST_DIR, 'images'))}")
print(f"Test labels: {count_files(os.path.join(TEST_DIR, 'labels'))}")

# %%
# 6. Check and update data.yaml file
print("\nChecking data.yaml...")
with open(YAML_PATH, 'r') as f:
    data_yaml = yaml.safe_load(f)
    print("Original data.yaml content:")
    print(data_yaml)

# Update paths in the data.yaml file
data_yaml['path'] = DATASET_DIR
data_yaml['train'] = 'train/images'
data_yaml['val'] = 'valid/images'  # Note: using 'valid' instead of 'validation'
data_yaml['test'] = 'test/images'

# Make sure class names are correct
if 'names' not in data_yaml or not data_yaml['names']:
    data_yaml['names'] = ['police_car']

with open(YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f)

print("\nUpdated data.yaml content:")
with open(YAML_PATH, 'r') as f:
    print(yaml.safe_load(f))

# %% [markdown]
# ## Training model with pretrained YOLOv8n model
# - 30 epochs
# - (640,640) image size (what yolo models accept)
# - 16 batch size
# - patience - 20, stops train if there is no trend change for 20 epochs.
# - freeze 10 first layers, good for new dataset on pretrained model.
# - learning rate - 0.001
# - warmup = 3, starts with low learning rate and increase later.
# - verbose = true, prints extended output during train.
# 

# %%
# 7. Train the YOLOv8 model
print("\nStarting model training...")

# Initialize the model with pre-trained weights
model = YOLO('yolov8n.pt')  # 'n' for nano model (smallest and fastest)

# Train the model
results = model.train(
    data=YAML_PATH,
    epochs=30,             # Total training epochs
    imgsz=640,             # Image size
    batch=16,              # Batch size
    patience=20,           # Early stopping patience
    freeze=10,             # Freeze first 10 layers
    lr0=0.001,             # Initial learning rate
    cos_lr=True,           # Use cosine LR scheduler
    warmup_epochs=3,       # Warmup epochs
    verbose=True,          # Verbose output
    plots=True,            # Generate plots
    save=True,             # Save results
)

# %% [markdown]
# ## Summary of Training Results - Explanation
# 
# This table presents the performance metrics of the YOLOv8 model across **all training epochs**. Each row corresponds to one epoch and shows how the model improves over time.
# 
# ---
# 
# ### What Do We See in the Table
# 
# - The model is trained for multiple epochs (iterations over the full dataset).
# - In each epoch, we log training losses and evaluation metrics.
# - At the end of training, we automatically highlight the **best epoch**, based on highest `mAP@50-95`, which reflects the model's overall detection performance.
# 
# ---
# 
# ### What Can We Learn
# 
# - A steady **decrease in losses** (Box Loss, Cls Loss, DFL Loss) indicates successful training.
# - **Precision**, **Recall**, and **mAP** values improve gradually, meaning the model learns to localize and classify objects better over time.
# - The best model (highlighted in green) is selected based on **maximum mAP@50-95**, indicating the best trade-off between localization and classification.
# 
# ---
# 
# ### Why `mAP@50-95` is Used to Select the Best Model
# 
# `mAP@50-95` is the most comprehensive and accepted metric for object detection performance.
# It averages the model's **precision and recall** across multiple IoU thresholds from 0.50 to 0.95 (step of 0.05), giving a balanced view of both **localization accuracy** and **classification quality**.
# 
# > The **best.pt** file saved by YOLO corresponds to the epoch with the highest `mAP@50-95` on the validation set.
# 
# ---
# 
# ### Column Descriptions
# 
# | Column        | Description                                                                                         | Notes / Formula |
# |---------------|-----------------------------------------------------------------------------------------------------|-----------------|
# | `Epoch`       | Training iteration number                                                                           | Starts at 0     |
# | `Box Loss`    | Measures error in bounding box regression                                                           | L1 + CIoU loss  |
# | `Cls Loss`    | Classification loss – how well the model predicts the correct class                                | Usually BCE loss|
# | `DFL Loss`    | Distribution Focal Loss – improves box localization resolution in dense regression                 | YOLOv8-specific |
# | `Precision`   | Of all predicted objects, how many are correct?                                                    | `TP / (TP + FP)`|
# | `Recall`      | Of all actual objects, how many were detected?                                                     | `TP / (TP + FN)`|
# | `mAP@50`      | Mean Average Precision at IoU=0.50 – measures object detection quality                             | Area under P-R curve |
# | `mAP@50-95`   | Mean of mAP across IoU thresholds from 0.50 to 0.95 (in steps of 0.05)                             | Final score to select best model |
# 
# ---
# 
# ### Definitions
# 
# - **TP** = True Positive (correct detection)  
# - **FP** = False Positive (wrong detection)  
# - **FN** = False Negative (missed object)  
# - **IoU** = Intersection over Union between predicted and ground truth boxes  

# %%
import os
import glob
import pandas as pd

# Find training folders that contain results.csv
train_dirs = sorted(
    [d for d in glob.glob('runs/detect/train*') if os.path.exists(os.path.join(d, 'results.csv'))],
    key=os.path.getmtime
)

if not train_dirs:
    raise FileNotFoundError("No valid training folder with results.csv was found.")

# Load the latest one
latest_train_dir = train_dirs[-1]
results_csv = os.path.join(latest_train_dir, 'results.csv')
df_results = pd.read_csv(results_csv)



# Extract relevant columns
summary_df = df_results[[
    'epoch',
    'train/box_loss',
    'train/cls_loss',
    'train/dfl_loss',
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)'
]]

# Rename columns for cleaner display
summary_df.columns = [
    'Epoch', 'Box Loss', 'Cls Loss', 'DFL Loss',
    'Precision', 'Recall', 'mAP@50', 'mAP@50-95'
]

# Identify best model row by highest mAP@50-95
best_idx = summary_df['mAP@50-95'].idxmax()

# Function to highlight best row in green
def highlight_best(row):
    return ['color: green; font-weight: bold;' if row.name == best_idx else '' for _ in row]

# Print title
print("\n\033[1mSummary of Training Results (All Epochs):\033[0m")
display(summary_df.style
        .apply(highlight_best, axis=1)
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
)

# Plot metrics across all epochs
plt.figure(figsize=(10, 5))
plt.plot(summary_df['Epoch'], summary_df['Precision'], marker='o', label='Precision')
plt.plot(summary_df['Epoch'], summary_df['Recall'], marker='o', label='Recall')
plt.plot(summary_df['Epoch'], summary_df['mAP@50'], marker='o', label='mAP@50')
plt.plot(summary_df['Epoch'], summary_df['mAP@50-95'], marker='o', label='mAP@50-95')
plt.title('Model Performance Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Best Model Summery

# %%
# Create dictionary with best model performance
val_metrics = {
    'Images': 108,
    'Instances': 112,
    'Precision': 0.998,
    'Recall': 0.982,
    'mAP@50': 0.994,
    'mAP@50-95': 0.867,
    'Preprocess Time (ms)': 0.4,
    'Inference Time (ms)': 2.4,
    'Postprocess Time (ms)': 4.5
}

# Convert to DataFrame for clean display
val_df = pd.DataFrame(val_metrics.items(), columns=['Metric', 'Value'])

# Display the summary
print("\n\033[1mBest Model Evaluation Summary:\033[0m")
display(val_df)

print(f"\nFinal model achieved {val_metrics['mAP@50-95']:.3f} mAP@50-95 with {val_metrics['Precision']:.1%} precision and {val_metrics['Recall']:.1%} recall.")

# %% [markdown]
# ## Final Model Validation and Export – Explanation
# 
# In this step, we perform the final evaluation of the trained YOLOv8 model and export it for deployment:
# 
# ---
# 
# ### What happens here?
# 
# 1. **Model Validation on the Test Set**
#    - The model is evaluated on 108 validation images.
#    - Key metrics:
#      - **Precision** = How many detections were correct (0.998)
#      - **Recall** = How many real objects were detected (0.982)
#      - **mAP@50** = Average precision at IoU threshold 0.5 (0.994)
#      - **mAP@50-95** = Mean AP across thresholds – overall score (0.868)
#    - These results indicate a **very high-quality model**.
# 
# 2. **Model Export**
#    - The trained model is exported to the ONNX format (`best.onnx`) for future deployment.
#    - Format: cross-platform and optimized for speed & compatibility.
# 
# 3. **Saving the Best Model**
#    - The model weights (`best.pt`) are saved to Google Drive at:
#      `/content/drive/MyDrive/israeli_police_model/best.pt`
# 
# ---
# 
# ### Why is this important?
# 
# - Final validation gives us a realistic understanding of model performance.
# - Exporting to ONNX makes the model usable in real-world apps.
# - Saving to Google Drive ensures we don’t lose our best model.
# 
# 

# %%
# Validate the model
print("\nValidating model on test set...")
metrics = model.val()
print(f"Model Performance Metrics:")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")

# Export the model
print("\nExporting model...")
model.export(format='onnx')

# Save to Google Drive
EXPORT_DIR = '/content/drive/MyDrive/israeli_police_model'
os.makedirs(EXPORT_DIR, exist_ok=True)

# Copy models to Google Drive
best_pt_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
if os.path.exists(best_pt_path):
    shutil.copy(best_pt_path, os.path.join(EXPORT_DIR, 'best.pt'))
    print(f"Best model saved to Google Drive at: {EXPORT_DIR}/best.pt")


# %%
# Define performance metrics from validation
val_metrics = {
    'Precision': 0.9981,
    'Recall': 0.9821,
    'mAP@50': 0.994,
    'mAP@50-95': 0.868
}

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(val_metrics.keys(), val_metrics.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.ylim(0, 1.05)
plt.title('Best Model Evaluation Metrics')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Testing the Model on Sample Images
# 
# In this section, we test our trained YOLOv8 model on **10 real-world sample images** from the test set.  
# The process includes **running inference**, **visualizing the predictions**, and then **summarizing the detection results** in a clean table.
# 
# ---
# 
# ### What Happens in This Section:
# 
# 1. The model runs inference on 10 images using `model.predict(...)`.
# 2. For each image, we visualize the **detected police car(s)** (if found) with bounding boxes and confidence scores.
# 3. We collect the **detection confidence** of the most confident detection in each image.
# 4. We display a summary table with the following info:
#    - Image file name  
#    - Whether any police car was detected  
#    - The confidence of the top detection
# 5. We also show a **summary row** that indicates:
#    - Total images with detection  
#    - Total without detection  
#    - Average confidence across all 10 images
# 
# ---
# 
# ### Why This Matters:
# 
# This part demonstrates the model's performance on real unseen images –
# highlighting not only **how well the model performs in terms of accuracy**, but also giving a **qualitative view** of the predictions.
# 
# It’s a crucial step in validating whether the model is ready for deployment or needs further tuning.
# 

# %%
# Test the model on multiple sample images
print("\nTesting on sample images...")
test_images_dir = os.path.join(TEST_DIR, 'images')
test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

results_list = []  # ← Save all model results for analysis in next cells

if test_images:
    for i, test_image in enumerate(test_images[:10]):
        test_image_path = os.path.join(test_images_dir, test_image)
        print(f"Testing image {i+1}/{min(10, len(test_images))}: {test_image}")

        # Run inference
        results = model(test_image_path)
        results_list.append(results)  # ← Store result for later use

        # Save and display result
        import cv2
        from google.colab.patches import cv2_imshow

        result_image = results[0].plot()
        result_path = f'/content/detection_result_{i+1}.jpg'
        cv2.imwrite(result_path, result_image)
        cv2_imshow(result_image)

        # Save to Google Drive
        shutil.copy(result_path, os.path.join(EXPORT_DIR, f'sample_detection_{i+1}.jpg'))

print("\nTesting complete!")

# %%
# Create detection summary table
results_data = []
for i, result in enumerate(results_list):
    boxes = result[0].boxes
    img_name = test_images[i]

    if boxes is not None and len(boxes) > 0:
        conf = boxes.conf.cpu().numpy().max()
        results_data.append({
            'Image': img_name,
            'Detected': '✅ Yes',
            'Confidence': round(conf, 2)
        })
    else:
        results_data.append({
            'Image': img_name,
            'Detected': '❌ No',
            'Confidence': 0.0
        })

# Convert to DataFrame
df = pd.DataFrame(results_data)

# Display the table
print("\n\033[1mDetection Summary for Sample Images:\033[0m")
display(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
    dict(selector='th', props=[('text-align', 'center')])
]))

# Show summary
avg_conf = df['Confidence'].mean()
num_yes = (df['Detected'] == '✅ Yes').sum()
num_no = (df['Detected'] == '❌ No').sum()

print(f"\n\033[1mSummary:\033[0m")
print(f"✅ Images with Detection: {num_yes}")
print(f"❌ Images without Detection: {num_no}")
print(f"📊 Average Confidence: {avg_conf:.2f}")



