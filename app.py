from scipy.ndimage import gaussian_filter
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image, ImageEnhance
import seaborn as sns
import numpy as np
import cv2
import random
import base64
from flask import Flask, request, jsonify, url_for
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras import layers, models
from PIL import Image
import tensorflow as tf

from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import traceback
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from flask import Flask, jsonify
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from flask import Flask, request, jsonify, send_file
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from flask import Flask, request, jsonify, url_for
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import io
from datetime import datetime, timedelta
import os
import csv
from collections import Counter
from flask import request, jsonify
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
# === Setup Flask App ===
app = Flask(__name__)  # initializing a web server that handles routes
CORS(app)  # make able to frontend-backend communication ,

LOG_FILE = 'log.csv'
# === Dataset Test Directory ===
DATASET_TEST_DIR = os.path.join('Dataset', 'test')


# # === Setup Flask App ===
# app = Flask(__name__)  # initializing a web server that handles routes
# CORS(app)  # make able to frontend-backend communication ,
#
# LOG_FILE = 'classification_log.csv'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
cnn = None  # global model variable


# Helper function to save and encode processed images
def save_and_encode_image(image, filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if isinstance(image, np.ndarray):
        # OpenCV image
        cv2.imwrite(file_path, image)
    else:
        # Pillow image
        # Convert RGBA -> RGB if necessary
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(file_path)

    # Encode image to Base64
    with open(file_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return encoded_string


# === Class Labels ===
CLASS_LABELS = ['mask_weared_incorrect', 'with_mask', 'without_mask']


if not os.path.exists('facemask_model.h5'):
    train_model()

# Load into the global variable 'cnn'
cnn = load_model('facemask_model.h5')
print("âœ… facemask_model.h5 loaded successfully.")
# === Save a prediction to the log (final version) ===
from datetime import datetime

# === Ensure log.csv exists ===
def ensure_log_file_integrity():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline="") as file :#newline prevents extra line spaces row wise
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Actual_Label']) #Writes a single row into the CSV file.

# === Get actual label from dataset ===
def get_actual_label_from_dataset(filename):
    for label in CLASS_LABELS:
        folder_path = os.path.join(DATASET_TEST_DIR, label)
        if not os.path.exists(folder_path):
            continue  # if folder_path does not exist then skip this and go back to for loop (for next loop iteration)
        if filename in os.listdir(folder_path):
            return label
    return "not_in_dataset"

# === Log a prediction ===
def log_prediction(filename, prediction):
    ensure_log_file_integrity()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    actual_label = get_actual_label_from_dataset(filename)

    corrected_prediction = prediction.strip().lower().replace('-', '_') #Replaces dashes with underscores (for consistency)
    # prediction is a string variable holding some text like "With-Mask",
    with open(LOG_FILE, mode='a', newline='') as file: # a measns adds new lines
        writer = csv.writer(file)
        writer.writerow([timestamp, filename, corrected_prediction, actual_label])

# === Predict Route ===
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Preprocess image
            image = Image.open(file_path).convert('RGB')
            image = image.resize((128, 128))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0) #it is mandatory to add a dimension before prediction at start otherwise error can come

            # Prediction
            prediction = cnn.predict(image_array)
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            # Log the prediction
            log_prediction(file.filename, predicted_class)

            return jsonify({
                "message": f"Prediction: {predicted_class} ({confidence:.2f}%)",
                "predicted_class": predicted_class,
                "confidence": f"{confidence:.2f}%"
            })

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"error": "No file uploaded"}), 400

# === Generate Report Route ===
@app.route('/generate_report', methods=['GET'])
def generate_report():
    try:
        ensure_log_file_integrity()
        today_str = datetime.now().strftime('%Y-%m-%d')
        filtered_rows = []

        with open(LOG_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Timestamp'].startswith(today_str):
                    filtered_rows.append(row)

        sorted_rows = sorted(filtered_rows, key=lambda x: datetime.strptime(x['Timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)

        corrected_preds = []
        for row in filtered_rows:
            pred = row['Prediction'].strip().lower().replace('-', '_')
            if pred in CLASS_LABELS:
                corrected_preds.append(pred)

        counts = Counter(corrected_preds)
        total = sum(counts.values())

        log_entries = []
        header = "| {:<20} | {:<25} | {:<20} | {:<20} |".format("Timestamp", "Filename", "Prediction", "Actual_Label")
        separator = "-" * len(header)
        log_entries.append(separator)
        log_entries.append(header)
        log_entries.append(separator)

        seen_files = set()
        unique_latest_rows = {}

        for row in sorted_rows:
            fname = row['Filename']
            if fname not in seen_files:
                unique_latest_rows[fname] = row
                seen_files.add(fname)

        unique_latest_rows = list(unique_latest_rows.values())[:5]

        if not unique_latest_rows:
            log_entries.append("| No entries found for today. |".center(len(separator)))
        else:
            for row in unique_latest_rows:
                log_entries.append(
                    "| {:<20} | {:<25} | {:<20} | {:<20} |".format(
                        row['Timestamp'], row['Filename'], row['Prediction'], row['Actual_Label']
                    )
                )

        log_entries.append(separator)

        return jsonify({
            'Date': datetime.now().strftime('%Y-%m-%d Time: %H:%M:%S'),
            'With Mask': counts.get('with_mask', 0),
            'Without Mask': counts.get('without_mask', 0),
            'Improperly Worn Mask': counts.get('mask_weared_incorrect', 0),
            'Total Classified': total,
            'Log': "\n".join(log_entries)
        })

    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    try:
        print("📢 Evaluate from log called!")

        if not os.path.exists(LOG_FILE):
            return jsonify({'error': 'Log file does not exist. Please make at least one prediction first.'}), 400

        with open(LOG_FILE, mode='r') as file:
            reader = csv.DictReader(file)

            rows = []
            today = datetime.today().date()

            for row in reader:
                timestamp_str = row.get('Timestamp')
                if not timestamp_str:
                    continue
                try:
                    row_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").date()
                    if row_date == today:
                        rows.append(row)
                except Exception:
                    continue

        if not rows:
            return jsonify({'error': 'No prediction records for today found in log file.'}), 400

        y_true = []
        y_pred = []

        for row in rows:
            actual = row.get('Actual_Label', '').strip().lower().replace('-', '_')
            pred = row.get('Prediction', '').strip().lower().replace('-', '_')

            if actual in CLASS_LABELS and pred in CLASS_LABELS:
                y_true.append(actual)
                y_pred.append(pred)

        if not y_true:
            return jsonify({'error': 'No valid prediction entries found in today\'s log.'}), 400

        label_to_index = {label: i for i, label in enumerate(CLASS_LABELS)}
        y_true_idx = [label_to_index[label] for label in y_true]
        y_pred_idx = [label_to_index[label] for label in y_pred]

        report = classification_report(
            y_true_idx, y_pred_idx, labels=list(range(len(CLASS_LABELS))),
            target_names=CLASS_LABELS, output_dict=True, zero_division=0
        )
        overall_accuracy = round(accuracy_score(y_true_idx, y_pred_idx), 4)

        metrics_per_class = {
            label: {
                'precision': round(report[label]['precision'], 3),
                'recall': round(report[label]['recall'], 3),
                'f1-score': round(report[label]['f1-score'], 3),
            }
            for label in CLASS_LABELS if label in report
        }

        y_true_one_hot = label_binarize(y_true_idx, classes=list(range(len(CLASS_LABELS))))
        y_pred_one_hot = label_binarize(y_pred_idx, classes=list(range(len(CLASS_LABELS))))

        auc_scores = {}
        plt.figure(figsize=(8, 6))
        for i in range(len(CLASS_LABELS)):
            if np.sum(y_true_one_hot[:, i]) == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_one_hot[:, i])
            auc = roc_auc_score(y_true_one_hot[:, i], y_pred_one_hot[:, i])
            auc_scores[CLASS_LABELS[i]] = round(auc, 3)
            plt.plot(fpr, tpr, label=f'{CLASS_LABELS[i]} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()

        if not os.path.exists('static'):
            os.makedirs('static')

        roc_path = 'static/roc_curve.png'
        plt.savefig(roc_path)
        plt.close()

        cm = confusion_matrix(y_true_idx, y_pred_idx)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = 'static/confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()

        actual_counts = Counter(y_true)
        predicted_counts = Counter(y_pred)
        x = np.arange(len(CLASS_LABELS))
        actual_vals = [actual_counts.get(label, 0) for label in CLASS_LABELS]
        predicted_vals = [predicted_counts.get(label, 0) for label in CLASS_LABELS]

        plt.figure(figsize=(8, 5))
        plt.bar(x - 0.2, actual_vals, width=0.4, label='Actual', color='skyblue')
        plt.bar(x + 0.2, predicted_vals, width=0.4, label='Predicted', color='salmon')
        plt.xticks(x, CLASS_LABELS)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Actual vs Predicted Class Distribution')
        plt.legend()
        plt.tight_layout()
        bar_path = 'static/actual_vs_predicted.png'
        plt.savefig(bar_path)
        plt.close()

        return jsonify({
            'overall_accuracy': overall_accuracy,
            'auc_scores': auc_scores,
            'metrics_per_class': metrics_per_class,
            'roc_curve_path': f'/{roc_path}',
            'confusion_matrix_path': f'/{cm_path}',
            'bar_chart_path': f'/{bar_path}'
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during evaluation: {str(e)}'}), 500



from flask import jsonify
import os, random, shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Paths
        original_dataset_dir = 'Dataset'
        train_dir = os.path.join(original_dataset_dir, 'train')
        val_dir = os.path.join(original_dataset_dir, 'val')
        test_dir = os.path.join(original_dataset_dir, 'test')

        # 1. Dataset split if needed
        train_empty = not os.listdir(train_dir) if os.path.exists(train_dir) else True
        val_empty = not os.listdir(val_dir) if os.path.exists(val_dir) else True
        test_empty = not os.listdir(test_dir) if os.path.exists(test_dir) else True

        if train_empty or val_empty or test_empty:
            print("Re-splitting dataset into train/val/test...")

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            for folder in os.listdir(original_dataset_dir):
                folder_path = os.path.join(original_dataset_dir, folder)
                if folder in ['train', 'val', 'test']:
                    continue

                image_files = os.listdir(folder_path)
                random.shuffle(image_files)
                total = len(image_files)
                train_split = int(total * 0.7)
                val_split = int(total * 0.85)

                train_files = image_files[:train_split]
                val_files = image_files[train_split:val_split]
                test_files = image_files[val_split:]

                os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
                os.makedirs(os.path.join(val_dir, folder), exist_ok=True)
                os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

                for file in train_files:
                    shutil.copy(os.path.join(folder_path, file), os.path.join(train_dir, folder, file))
                for file in val_files:
                    shutil.copy(os.path.join(folder_path, file), os.path.join(val_dir, folder, file))
                for file in test_files:
                    shutil.copy(os.path.join(folder_path, file), os.path.join(test_dir, folder, file))

            print("✅ Dataset split complete: 70% train, 15% val, 15% test")
        else:
            print("✅ Dataset already split. No changes made.")
        train_datagen = ImageDataGenerator( # augmentation for better model training
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1. / 255) #model itself testing
        test_datagen = ImageDataGenerator(rescale=1. / 255) # maha testing

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=16,
            class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(128, 128),
            batch_size=16,
            class_mode='categorical'
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(128, 128),
            batch_size=16, #32 images at once
            class_mode='categorical'
        )

        # 3. Model Architecture
        def create_cnn(input_shape=(128, 128, 3), num_classes=3):
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=input_shape))# Detects features basic features (edges)
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))# mid-level patterns (shapes, curves)
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))# Detects face components and mask position, Complex objects (e.g., eyes, masks, faces)
            model.add(layers.MaxPooling2D((2, 2))) #MaxPooling: reduces spatial(spatial refers to the width and height dimensions) size and keeps the most important features.
            model.add(layers.Flatten()) # for dense use because dense cant use height and weight separately
            model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))) #  deep pattern recognition (makes a first guess)
            model.add(layers.Dropout(0.6)) #Dropout “reminds” the network to not depend too much on any neuron, making the learned features more robust and generalizable.
            #The dropped neurons only skip this one training step. In the next training step (next batch), dropout will randomly choose a (different) set of neurons to drop.
            model.add(layers.Dense(128, activation='relu'))      # Refines decision-making from the learned robust features after dropout

            model.add(layers.Dense(num_classes, activation='softmax')) # softmax is for multitasks and it comes at last output layer ,# Final classification layer: outputs probability of each mask class
            return model
             #kernel_regularizer. This is a parameter used in certain Keras layers
        #.Regularization helps prevent overfitting by adding a penalty(cost). 	kernel_regularizer= kernel refers to weights and regulazier is for smootthing (only for understanding)
        #l2(0.001) adds a small penalty on large weights, which helps update the weights in a way that discourages large values and avoid overfitting and generalize better.
        cnn = create_cnn()
        cnn.compile(  #A function from the Keras Model class that prepares the model
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  #Optimizer = the algorithm that adjusts model weights during training based on the error (loss). lr according to per batch
            loss='categorical_crossentropy', # this is a type of loss in keras
            metrics=['accuracy']
        )
#The learning rate controls how quickly or slowly the model updates its weights based on the error (loss)
        # 4. Model Training
        history = cnn.fit(      #fitThis is a method of the Keras model.
            train_generator,
            epochs=20, # model training numbers
            validation_data=val_generator #It helps the function understand what that argument is for
        )

        # 5. Save model
        cnn.save('facemask_model.h5')
        print("✅ Model training completed and saved as facemask_model.h5")

        # ✅ Response to client
        return jsonify({'message': '✅ Retraining completed!'})

    except Exception as e:
        print("❌ Error during training:", str(e))
        return jsonify({'message': '❌ Failed to retrain model.', 'error': str(e)}), 500



# Route to serve HTML
@app.route('/')
def index():
    return render_template('index.html')


### -------------------- PREPROCESSING --------------------

# Image Normalization
@app.route('/normalize', methods=['POST'])
def normalize():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            encoded_image = save_and_encode_image(normalized_image, 'normalized_image.jpg')
            return jsonify({"message": "Image normalized successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Normalization failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Noise Reduction (using Gaussian Blur)
@app.route('/noise_reduction', methods=['POST'])
def noise_reduction():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
            encoded_image = save_and_encode_image(denoised_image, 'noise_reduced.jpg')
            return jsonify({"message": "Noise reduced successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Noise reduction failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Artifact Removal (using Bilateral Filter)
@app.route('/artifact_removal', methods=['POST'])
def artifact_removal():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            artifact_removed_image = cv2.bilateralFilter(image, 9, 75, 75)
            encoded_image = save_and_encode_image(artifact_removed_image, 'artifact_removed.jpg')
            return jsonify({"message": "Artifact removal applied successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Artifact removal failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


### -------------------- DATA AUGMENTATION --------------------

# Rotation
@app.route('/rotate', methods=['POST'])
def rotate():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = Image.open(file_path)
            rotated_image = image.rotate(90)
            encoded_image = save_and_encode_image(rotated_image, 'rotated_image.jpg')
            return jsonify({"message": "Image rotated successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Rotation failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Random Cropping
@app.route('/random_crop', methods=['POST'])
def random_crop():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = Image.open(file_path)
            width, height = image.size
            new_width, new_height = int(width * 0.8), int(height * 0.8)
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            cropped_image = image.crop((left, top, left + new_width, top + new_height))
            encoded_image = save_and_encode_image(cropped_image, 'random_cropped.jpg')
            return jsonify({"message": "Random cropping applied successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Random cropping failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Flipping (renamed to match the HTML form action '/flipping')
@app.route('/flipping', methods=['POST'])
def flip():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = Image.open(file_path)
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            encoded_image = save_and_encode_image(flipped_image, 'flipped_image.jpg')
            return jsonify({"message": "Image flipped successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Flipping failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Scaling
@app.route('/scaling', methods=['POST'])
def scale():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = Image.open(file_path)
            scaled_image = image.resize((int(image.width * 1.5), int(image.height * 1.5)))
            encoded_image = save_and_encode_image(scaled_image, 'scaled_image.jpg')
            return jsonify({"message": "Image scaled successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Scaling failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Noise Injection
@app.route('/noise_injection', methods=['POST'])
def noise_injection():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
            encoded_image = save_and_encode_image(noisy_image, 'noisy_image.jpg')
            return jsonify({"message": "Noise injected successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Noise injection failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Translation
@app.route('/translation', methods=['POST'])
def translation():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            tx, ty = 50, 50  # Translation values
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
            encoded_image = save_and_encode_image(translated_image, 'translated_image.jpg')
            return jsonify({"message": "Image translated successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Elastic Deformation
def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return distorted_image


@app.route('/elastic_deformation', methods=['POST'])
def elastic_deformation():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            alpha, sigma = 34, 4  # Adjust values as needed
            deformed_image = elastic_transform(image, alpha, sigma)
            encoded_image = save_and_encode_image(deformed_image, 'elastic_deformation.jpg')
            return jsonify({"message": "Elastic deformation applied successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Elastic deformation failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Intensity Adjustment
@app.route('/intensity_adjustment', methods=['POST'])
def intensity_adjustment():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = Image.open(file_path)
            enhancer = ImageEnhance.Brightness(image)
            adjusted_image = enhancer.enhance(1.5)
            encoded_image = save_and_encode_image(adjusted_image, 'intensity_adjusted.jpg')
            return jsonify({"message": "Intensity adjusted successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Intensity adjustment failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Shearing
@app.route('/shearing', methods=['POST'])
def shearing():
    file = request.files.get('file')
    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file.")
            height, width = image.shape[:2]
            shear_x = 0.2  # Shearing along x-axis
            shear_y = 0.0  # Shearing along y-axis
            M = np.array([[1, shear_x, 0],
                          [shear_y, 1, 0]], dtype=np.float32)
            sheared_image = cv2.warpAffine(image, M, (width, height))
            encoded_image = save_and_encode_image(sheared_image, 'sheared_image.jpg')
            return jsonify({"message": "Shearing applied successfully!", "encoded_image": encoded_image})
        except Exception as e:
            return jsonify({"error": f"Shearing failed: {str(e)}"}), 500
    return jsonify({"error": "No file uploaded"}), 400


# Serve Processed Images
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
