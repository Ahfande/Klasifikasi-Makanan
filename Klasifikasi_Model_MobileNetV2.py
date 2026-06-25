# ======================
# 1. IMPORT LIBRARY
# ======================
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from matplotlib.patches import Rectangle

np.random.seed(42)
tf.random.set_seed(42)

# ======================
# 2. MOUNT GOOGLE DRIVE DAN LOAD DATASET
# ======================
from google.colab import drive
drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/skripsi/dataset/"
print(f"Base path: {base_path}")

# ======================
# 3. VALIDASI DATASET SECARA LANJUTAN
# ======================
def advanced_dataset_validation(base_path):
    """Validasi dataset secara mendalam"""
    print("="*60)
    print("ADVANCED DATASET VALIDATION")
    print("="*60)

    splits = ['train', 'val', 'test']
    dataset_stats = {}

    for split in splits:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"❌ ERROR: Folder {split} tidak ditemukan!")
            continue

        classes = sorted([d for d in os.listdir(split_path)
                         if os.path.isdir(os.path.join(split_path, d))])

        split_stats = {
            'total_images': 0,
            'class_distribution': {},
            'image_sizes': [],
            'valid_images': 0
        }

        print(f"\n📊 {split.upper()} Statistics:")
        print("-" * 40)

        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            image_files = [f for f in os.listdir(cls_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            valid_images = []
            for img_file in image_files:
                img_path = os.path.join(cls_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        valid_images.append(img_path)
                        split_stats['image_sizes'].append(img.shape[:2])
                except:
                    continue

            split_stats['class_distribution'][cls] = len(valid_images)
            split_stats['total_images'] += len(valid_images)
            split_stats['valid_images'] += len(valid_images)

            print(f"  {cls}: {len(valid_images)} gambar")

        dataset_stats[split] = split_stats

        print(f"\n  Total valid images: {split_stats['valid_images']}")
        print(f"  Avg image size: {np.mean(split_stats['image_sizes'], axis=0) if split_stats['image_sizes'] else 'N/A'}")

    return dataset_stats
dataset_stats = advanced_dataset_validation(base_path)

# ======================
# 4. PREPROCESSING DAN AUGMENTASI DATA
# ======================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Changed to MobileNetV2 preprocessing

IMG_SIZE = 224 
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

print("\n" + "="*60)
print("CREATING DATA GENERATORS")
print("="*60)

train_data = train_datagen.flow_from_directory(
    os.path.join(base_path, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_data = val_datagen.flow_from_directory(
    os.path.join(base_path, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

test_data = test_datagen.flow_from_directory(
    os.path.join(base_path, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)

print(f"\nData loaded successfully!")
print(f"   Number of classes: {num_classes}")
print(f"   Class names: {class_names}")
print(f"   Training samples: {train_data.samples}")
print(f"   Validation samples: {val_data.samples}")
print(f"   Test samples: {test_data.samples}")

# ======================
# 5. VISUALISASI DATA HASIL AUGMENTASI
# ======================
def visualize_augmented_data(generator, num_images=8):
    """Visualisasi gambar hasil augmentasi"""
    images, labels = next(generator)

    plt.figure(figsize=(15, 8))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 4, i+1)

        img = images[i]
        img = img - img.min()
        img = img / img.max()

        plt.imshow(img)

        label_idx = np.argmax(labels[i])
        class_name = class_names[label_idx]

        plt.title(f"{class_name}", fontsize=12)
        plt.axis('off')

    plt.suptitle("Augmented Training Images", fontsize=16)
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("VISUALIZING AUGMENTED DATA")
print("="*60)
visualize_augmented_data(train_data)

# ======================
# 6. MEMBANGUN MODEL MOBILENETV2
# ======================
print("\n" + "="*60)
print("BUILDING MOBILENETV2 MODEL")
print("="*60)

from tensorflow.keras.applications import MobileNetV2  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

tf.keras.backend.clear_session()

print("Loading MobileNetV2 with ImageNet weights...")
base_model = MobileNetV2(  
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Layer dense pertama dengan regularisasi
x = Dense(512, activation='relu',
          kernel_regularizer=l2(0.001),
          bias_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Layer dense kedua
x = Dense(256, activation='relu',
          kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Layer output
outputs = Dense(num_classes, activation='softmax')(x)

# Membuat model
model = Model(inputs, outputs)

# Kompilasi model dengan learning rate custom
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# ======================
# 7. CALLBACK UNTUK TRAINING
# ======================
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1,
        min_delta=0.001
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    ),

    ModelCheckpoint(
        filepath='models/best_mobilenetv2_model.h5',  
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    ),

    TensorBoard(
        log_dir='logs/fit_mobilenetv2',  
        histogram_freq=1,
        update_freq='epoch'
    ),

    CSVLogger('training_log_mobilenetv2.csv')  
]

# ======================
# 8. STRATEGI TRAINING DUA FASE
# ======================
print("\n" + "="*60)
print("TWO-PHASE TRAINING STRATEGY")
print("="*60)

# FASE 1: Feature Extraction
print("\n PHASE 1: Feature Extraction (Frozen Base Model)")
print("-" * 50)

train_steps = len(train_data)
val_steps = len(val_data)

print(f"Train steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")

history_phase1 = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

# FASE 2: Fine-tuning
print("\n PHASE 2: Fine-tuning (Unfreeze Some Layers)")
print("-" * 50)

base_model.trainable = True

total_layers = len(base_model.layers)
fine_tune_at = total_layers - 50 

print(f"Total base model layers: {total_layers}")
print(f"Freezing layers 0-{fine_tune_at-1}")
print(f"Fine-tuning layers {fine_tune_at}-{total_layers-1} (total {total_layers - fine_tune_at} layers)")

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

history_phase2 = model.fit(
    train_data,
    initial_epoch=history_phase1.epoch[-1],
    epochs=35,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

# ======================
# 9. EVALUASI MODEL SECARA KOMPREHENSIF
# ======================
print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

best_model = tf.keras.models.load_model('models/best_mobilenetv2_model.h5')

print("\n📊 Final Evaluation on Test Set:")
results = best_model.evaluate(test_data, verbose=0)

print(f"✅ Test Loss: {results[0]:.4f}")
print(f"✅ Test Accuracy: {results[1]:.2%}")
print(f"✅ Test Precision: {results[2]:.2%}")
print(f"✅ Test Recall: {results[3]:.2%}")
print(f"✅ F1-Score: {(2 * results[2] * results[3]) / (results[2] + results[3] + 1e-7):.2%}")

print("\n🔍 Generating predictions and metrics...")
y_pred = best_model.predict(test_data, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

# ======================
# 10. VISUALISASI HASIL TRAINING
# ======================
def plot_advanced_training_history(history1, history2):
    """Plot history training dengan dua fase"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    total_epochs = len(history1.history['accuracy']) + len(history2.history['accuracy'])
    epochs_range = range(1, total_epochs + 1)

    # Accuracy
    axes[0, 0].plot(epochs_range[:len(history1.history['accuracy'])],
                   history1.history['accuracy'], 'b-', label='Phase 1 Train', linewidth=2)
    axes[0, 0].plot(epochs_range[:len(history1.history['val_accuracy'])],
                   history1.history['val_accuracy'], 'r-', label='Phase 1 Val', linewidth=2)
    axes[0, 0].plot(epochs_range[len(history1.history['accuracy']):],
                   history2.history['accuracy'], 'b--', label='Phase 2 Train', linewidth=2)
    axes[0, 0].plot(epochs_range[len(history1.history['val_accuracy']):],
                   history2.history['val_accuracy'], 'r--', label='Phase 2 Val', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(epochs_range[:len(history1.history['loss'])],
                   history1.history['loss'], 'b-', label='Phase 1 Train', linewidth=2)
    axes[0, 1].plot(epochs_range[:len(history1.history['val_loss'])],
                   history1.history['val_loss'], 'r-', label='Phase 1 Val', linewidth=2)
    axes[0, 1].plot(epochs_range[len(history1.history['loss']):],
                   history2.history['loss'], 'b--', label='Phase 2 Train', linewidth=2)
    axes[0, 1].plot(epochs_range[len(history1.history['val_loss']):],
                   history2.history['val_loss'], 'r--', label='Phase 2 Val', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[0, 2].plot(epochs_range[:len(history1.history.get('precision', []))],
                   history1.history.get('precision', []), 'g-', label='Train Precision', linewidth=2)
    axes[0, 2].plot(epochs_range[:len(history1.history.get('val_precision', []))],
                   history1.history.get('val_precision', []), 'm-', label='Val Precision', linewidth=2)
    axes[0, 2].set_title('Model Precision', fontsize=14)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    if 'lr' in history1.history:
        axes[1, 0].plot(epochs_range[:len(history1.history['lr'])],
                       history1.history['lr'], 'purple', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')

    train_acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    axes[1, 2].plot(epochs_range, train_acc, 'b-', label='Training', linewidth=2)
    axes[1, 2].plot(epochs_range, val_acc, 'r-', label='Validation', linewidth=2)
    axes[1, 2].fill_between(epochs_range, train_acc, val_acc, alpha=0.2)
    axes[1, 2].set_title('Training vs Validation Accuracy', fontsize=14)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Advanced Training Analytics - MobileNetV2', fontsize=16)
    plt.tight_layout()
    plt.show()

print("\n📋 Detailed Classification Report:")
report = classification_report(y_true, y_pred_classes,
                               target_names=class_names,
                               output_dict=True)

report_df = pd.DataFrame(report).transpose()
print(report_df.to_string())

def plot_enhanced_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix yang ditingkatkan"""
    cm = confusion_matrix(y_true, y_pred)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('Model Confusion Analysis - MobileNetV2', fontsize=16)
    plt.tight_layout()
    plt.show()

    return cm

print("\n📈 Enhanced Confusion Matrix:")
cm = plot_enhanced_confusion_matrix(y_true, y_pred_classes, class_names)

# ======================
# 11. ANALISIS PERFORMANCE PER KELAS
# ======================
def analyze_class_performance(y_true, y_pred, class_names):
    """Analisis performa per kelas"""
    class_stats = {}

    for i, class_name in enumerate(class_names):
        idx = (y_true == i)

        if np.sum(idx) > 0:
            correct = np.sum(y_pred[idx] == i)
            total = np.sum(idx)
            accuracy = correct / total

            pred_for_class = y_pred == i
            precision = correct / np.sum(pred_for_class) if np.sum(pred_for_class) > 0 else 0

            class_stats[class_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'samples': total,
                'correct': correct
            }

    stats_df = pd.DataFrame.from_dict(class_stats, orient='index')
    stats_df = stats_df.sort_values('accuracy', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].barh(range(len(stats_df)), stats_df['accuracy'], color='skyblue')
    axes[0, 0].set_yticks(range(len(stats_df)))
    axes[0, 0].set_yticklabels(stats_df.index)
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Accuracy per Class', fontsize=14)
    axes[0, 0].invert_yaxis()

    axes[0, 1].barh(range(len(stats_df)), stats_df['precision'], color='lightgreen')
    axes[0, 1].set_yticks(range(len(stats_df)))
    axes[0, 1].set_yticklabels(stats_df.index)
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_title('Precision per Class', fontsize=14)
    axes[0, 1].invert_yaxis()

    axes[1, 0].barh(range(len(stats_df)), stats_df['samples'], color='lightcoral')
    axes[1, 0].set_yticks(range(len(stats_df)))
    axes[1, 0].set_yticklabels(stats_df.index)
    axes[1, 0].set_xlabel('Number of Samples')
    axes[1, 0].set_title('Samples per Class', fontsize=14)
    axes[1, 0].invert_yaxis()

    axes[1, 1].scatter(stats_df['samples'], stats_df['accuracy'], s=100, alpha=0.6)
    for i, txt in enumerate(stats_df.index):
        axes[1, 1].annotate(txt, (stats_df['samples'].iloc[i], stats_df['accuracy'].iloc[i]),
                           fontsize=9, alpha=0.8)
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Sample Size', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Class-wise Performance Analysis - MobileNetV2', fontsize=16)
    plt.tight_layout()
    plt.show()

    return stats_df

print("\n📊 Class-wise Performance Analysis:")
class_stats_df = analyze_class_performance(y_true, y_pred_classes, class_names)
print("\nClass Statistics:")
print(class_stats_df.to_string())

# ======================
# 12. FUNGSI PREDIKSI UNTUK GAMBAR CUSTOM
# ======================
def predict_with_confidence(model, image_path, class_names, top_n=3):
    """Prediksi gambar dengan confidence score"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = preprocess_input(np.expand_dims(img_resized, axis=0))

    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = predictions[top_indices]

    return {
        'top_class': top_classes[0],
        'top_confidence': top_confidences[0],
        'all_predictions': list(zip(top_classes, top_confidences)),
        'original_image': img_rgb,
        'processed_image': img_resized
    }

def visualize_prediction(results, class_names):
    """Visualisasi hasil prediksi"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(results['original_image'])
    axes[0].set_title(f"Original Image\nPredicted: {results['top_class']}", fontsize=14)
    axes[0].axis('off')

    top_n = len(results['all_predictions'])
    classes, confidences = zip(*results['all_predictions'])

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = axes[1].barh(range(top_n), confidences, color=colors)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_title(f'Top {top_n} Predictions', fontsize=14)
    axes[1].invert_yaxis()
    axes[1].set_xlim([0, 1])

    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        width = bar.get_width()
        axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{conf:.2%}', va='center', fontsize=10, fontweight='bold')

    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(2)

    plt.suptitle(f"🎯 Food Detection Result: {results['top_class']} ({results['top_confidence']:.2%})",
                fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n{'='*50}")
    print(f"🔍 PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Top Prediction: {results['top_class']} ({results['top_confidence']:.2%})")
    print(f"\nTop {len(results['all_predictions'])} Predictions:")
    for i, (cls, conf) in enumerate(results['all_predictions'], 1):
        print(f"  {i}. {cls}: {conf:.2%}")

# ======================
# 13. UJI COBA DENGAN GAMBAR CUSTOM
# ======================
print("\n" + "="*60)
print("TESTING WITH CUSTOM IMAGES")
print("="*60)


print("📤 Upload images for testing (max 3 images):")
uploaded = files.upload()

for filename in list(uploaded.keys())[:3]:
    print(f"\n{'='*50}")
    print(f"📷 File: {filename}")
    print('='*50)

    file_path = f"/content/{filename}"

    results = predict_with_confidence(best_model, file_path, class_names, top_n=3)

    if results:
        visualize_prediction(results, class_names)

# ======================
# 14. MENYIMPAN MODEL DAN FILE PENTING
# ======================
print("\n" + "="*60)
print("SIMPLE MODEL SAVING")
print("="*60)

print("\n💾 Saving model and essential files...")

try:
    best_model.save('deteksi_makanan_mobilenetv2.keras')
    print("✅ 1. Model saved: deteksi_makanan_mobilenetv2.keras")
except Exception as e:
    print(f"❌ Error saving .keras: {e}")

try:
    best_model.save('deteksi_makanan_mobilenetv2.h5')
    print("✅ 2. Model saved: deteksi_makanan_mobilenetv2.h5")
except Exception as e:
    print(f"❌ Error saving .h5: {e}")

try:
    best_model.save_weights('deteksi_makanan_mobilenetv2.weights.h5')
    print("✅ 3. Model weights saved: deteksi_makanan_mobilenetv2.weights.h5")
except Exception as e:
    print(f"⚠️ Error saving weights: {e}")

print("\n📊 Evaluating model to get metrics...")
evaluation_results = best_model.evaluate(test_data, verbose=0)

print(f"Type of evaluation_results: {type(evaluation_results)}")
print(f"Evaluation results: {evaluation_results}")

if isinstance(evaluation_results, dict):
    test_loss = evaluation_results.get('loss', 0)
    test_accuracy = evaluation_results.get('accuracy', 0)
    test_precision = evaluation_results.get('precision', 0)
    test_recall = evaluation_results.get('recall', 0)
else:
    try:
        test_loss = evaluation_results[0] if len(evaluation_results) > 0 else 0
        test_accuracy = evaluation_results[1] if len(evaluation_results) > 1 else 0
        test_precision = evaluation_results[2] if len(evaluation_results) > 2 else 0
        test_recall = evaluation_results[3] if len(evaluation_results) > 3 else 0
    except:
        test_loss = test_accuracy = test_precision = test_recall = 0

# Hitung F1-score
if (test_precision + test_recall) > 0:
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)
else:
    test_f1 = 0

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Test Precision: {test_precision:.2%}")
print(f"Test Recall: {test_recall:.2%}")
print(f"Test F1-Score: {test_f1:.2%}")
print("\nSaving class information...")

class_info = {
    'class_names': class_names,
    'class_indices': train_data.class_indices,
    'num_classes': num_classes,
    'img_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'model_name': 'MobileNetV2',
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'test_f1_score': float(test_f1),
    'training_date': '4 Maret 2026'
}

with open('model_info_mobilenetv2.json', 'w') as f:
    json.dump(class_info, f, indent=4)
print("✅ 4. Model info saved: model_info_mobilenetv2.json")

print("\n📊 Saving evaluation data...")

try:
    y_pred = best_model.predict(test_data, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_data.classes

    cm = confusion_matrix(y_true, y_pred_classes)

    np.save('confusion_matrix_mobilenetv2.npy', cm)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv('confusion_matrix_mobilenetv2.csv')

    print("✅ 5. Confusion matrix saved: confusion_matrix_mobilenetv2.npy & .csv")

except Exception as e:
    print(f"⚠️ Error saving confusion matrix: {e}")

print("\n📝 Saving training history...")

try:
    if 'history_phase1' in locals():
        history_data = {}

        if hasattr(history_phase1, 'history'):
            history_data['phase1'] = {}
            for key, values in history_phase1.history.items():
                history_data['phase1'][key] = [float(v) for v in values]

        if 'history_phase2' in locals() and hasattr(history_phase2, 'history'):
            history_data['phase2'] = {}
            for key, values in history_phase2.history.items():
                history_data['phase2'][key] = [float(v) for v in values]

        with open('training_history_mobilenetv2.json', 'w') as f:
            json.dump(history_data, f, indent=4)

        print("✅ 6. Training history saved: training_history_mobilenetv2.json")

except Exception as e:
    print(f"⚠️ Error saving history: {e}")

print("\n🎨 Creating summary visualization...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if 'history_phase1' in locals() and hasattr(history_phase1, 'history'):
        axes[0, 0].plot(history_phase1.history.get('accuracy', []), label='Train Phase 1', linewidth=2)
        axes[0, 0].plot(history_phase1.history.get('val_accuracy', []), label='Validation Phase 1', linewidth=2)
        if 'history_phase2' in locals():
            axes[0, 0].plot(history_phase2.history.get('accuracy', []), label='Train Phase 2', linewidth=2, linestyle='--')
            axes[0, 0].plot(history_phase2.history.get('val_accuracy', []), label='Validation Phase 2', linewidth=2, linestyle='--')
        axes[0, 0].set_title('Training Accuracy - MobileNetV2')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    if 'cm' in locals():
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix - MobileNetV2')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('True')
        axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 0].barh(class_names, [test_accuracy] * len(class_names), color='skyblue')
    axes[1, 0].set_title(f'Model Accuracy: {test_accuracy:.2%}')
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_xlim([0, 1])

    axes[1, 1].axis('off')
    summary_text = f"""
    MODEL SUMMARY - MobileNetV2
    ===========================
    Architecture: MobileNetV2

    Test Accuracy: {test_accuracy:.2%}
    Test Precision: {test_precision:.2%}
    Test Recall: {test_recall:.2%}
    F1-Score: {test_f1:.2%}
    Test Loss: {test_loss:.4f}
    Classes: {num_classes}
    Image Size: {IMG_SIZE}x{IMG_SIZE}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                   verticalalignment='center', fontfamily='monospace')

    plt.suptitle('Food Detection Model - MobileNetV2 Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_summary_mobilenetv2.png', dpi=300, bbox_inches='tight')
    print("✅ 7. Model summary saved: model_summary_mobilenetv2.png")

    plt.show()

except Exception as e:
    print(f"⚠️ Error creating visualization: {e}")

# ======================
# 15. CEK DIMENSI DATA SEBELUM GRAFIK
# ======================
print("\n📊 Cek dimensi data...")
print(f"Phase 1 - accuracy: {len(history_phase1.history['accuracy'])} data")
print(f"Phase 1 - val_accuracy: {len(history_phase1.history['val_accuracy'])} data")
print(f"Phase 2 - accuracy: {len(history_phase2.history['accuracy'])} data")
print(f"Phase 2 - val_accuracy: {len(history_phase2.history['val_accuracy'])} data")

len_phase1 = len(history_phase1.history['accuracy'])
len_phase2 = len(history_phase2.history['accuracy'])

epochs_phase1 = range(1, len_phase1 + 1)  # 1 sampai 20
epochs_phase2 = range(20, 20 + len_phase2)  # 20 sampai 35

print(f"\n✅ Epoch Phase 1: {min(epochs_phase1)} - {max(epochs_phase1)}")
print(f"✅ Epoch Phase 2: {min(epochs_phase2)} - {max(epochs_phase2)}")

# ======================
# 16. GRAFIK AKURASI FASE 1
# ======================
plt.figure(figsize=(12, 7))

plt.plot(epochs_phase1, history_phase1.history['accuracy'],
         'b-o', label='Train Accuracy', linewidth=2, markersize=6)
plt.plot(epochs_phase1, history_phase1.history['val_accuracy'],
         'r-s', label='Validation Accuracy', linewidth=2, markersize=6)

best_epoch_phase1 = np.argmax(history_phase1.history['val_accuracy']) + 1
best_val_phase1 = max(history_phase1.history['val_accuracy'])

plt.scatter(best_epoch_phase1, best_val_phase1,
            color='green', s=200, zorder=5, edgecolors='black', linewidth=2)
plt.annotate(f'Best: {best_val_phase1:.2%}\nEpoch {best_epoch_phase1}',
             xy=(best_epoch_phase1, best_val_phase1),
             xytext=(best_epoch_phase1 + 2, best_val_phase1 - 0.03),
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.title('Phase 1: Feature Extraction (Model Base Dikunci)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.40, 0.95)
plt.xticks(range(0, len_phase1 + 1, 2))
plt.tight_layout()
plt.savefig('grafik_bab4_fase1_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   ✅ Fase 1: Best validation accuracy = {best_val_phase1:.2%} at epoch {best_epoch_phase1}")

# ======================
# 17. GRAFIK AKURASI FASE 2
# ======================
plt.figure(figsize=(12, 7))

plt.plot(epochs_phase2, history_phase2.history['accuracy'],
         'b--o', label='Train Accuracy', linewidth=2, markersize=6)
plt.plot(epochs_phase2, history_phase2.history['val_accuracy'],
         'r--s', label='Validation Accuracy', linewidth=2, markersize=6)

val_acc_phase2 = history_phase2.history['val_accuracy']
best_epoch_in_phase2 = np.argmax(val_acc_phase2)
best_epoch_phase2 = list(epochs_phase2)[best_epoch_in_phase2]
best_val_phase2 = max(val_acc_phase2)

plt.scatter(best_epoch_phase2, best_val_phase2,
            color='green', s=200, zorder=5, edgecolors='black', linewidth=2)
plt.annotate(f'Best: {best_val_phase2:.2%}\nEpoch {best_epoch_phase2}',
             xy=(best_epoch_phase2, best_val_phase2),
             xytext=(best_epoch_phase2 - 3, best_val_phase2 - 0.03),
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.title('Phase 2: Fine-Tuning (Beberapa Layer Dibuka)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.85, 0.99)
plt.xticks(list(epochs_phase2)[::2])
plt.tight_layout()
plt.savefig('grafik_bab4_fase2_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   ✅ Fase 2: Best validation accuracy = {best_val_phase2:.2%} at epoch {best_epoch_phase2}")
print(f"   📈 Peningkatan: +{(best_val_phase2 - best_val_phase1)*100:.2f}% dari fase 1")

# ======================
# 18. GRAFIK LOSS FASE 1 & FASE 2
# ======================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(epochs_phase1, history_phase1.history['loss'],
         'b-o', label='Train Loss', linewidth=2, markersize=5)
ax1.plot(epochs_phase1, history_phase1.history['val_loss'],
         'r-s', label='Validation Loss', linewidth=2, markersize=5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Phase 1: Loss (Feature Extraction)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_phase2, history_phase2.history['loss'],
         'b--o', label='Train Loss', linewidth=2, markersize=5)
ax2.plot(epochs_phase2, history_phase2.history['val_loss'],
         'r--s', label='Validation Loss', linewidth=2, markersize=5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Phase 2: Loss (Fine-Tuning)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafik_bab4_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 19. GRAFIK GABUNGAN (1-35)
# ======================
plt.figure(figsize=(14, 7))

train_acc_full = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc_full = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
epochs_full = list(epochs_phase1) + list(epochs_phase2)

split_idx = len(epochs_phase1) - 1
plt.plot(epochs_full[:split_idx+1], train_acc_full[:split_idx+1], 'b-', linewidth=2, label='Train (Phase 1)')
plt.plot(epochs_full[:split_idx+1], val_acc_full[:split_idx+1], 'r-', linewidth=2, label='Validation (Phase 1)')
plt.plot(epochs_full[split_idx:], train_acc_full[split_idx:], 'b--', linewidth=2, label='Train (Phase 2)')
plt.plot(epochs_full[split_idx:], val_acc_full[split_idx:], 'r--', linewidth=2, label='Validation (Phase 2)')

plt.axvline(x=20, color='purple', linestyle='-', linewidth=2, alpha=0.7)
plt.text(20.5, 0.65, 'Start Fine-Tuning', rotation=90, fontsize=10, color='purple')

plt.scatter(best_epoch_phase1, best_val_phase1, color='green', s=150, zorder=5)
plt.scatter(best_epoch_phase2, best_val_phase2, color='darkgreen', s=150, zorder=5)

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.title('Complete Training: Feature Extraction vs Fine-Tuning', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0.40, 1.00)
plt.tight_layout()
plt.savefig('grafik_bab4_accuracy_gabungan.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 20. GRAFIK GAP OVERFITTING
# ======================
plt.figure(figsize=(14, 6))

train_acc_full = np.array(history_phase1.history['accuracy'] + history_phase2.history['accuracy'])
val_acc_full = np.array(history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'])
gap = train_acc_full - val_acc_full

colors = ['red' if g > 0 else 'green' for g in gap]
plt.bar(epochs_full, gap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

plt.axvspan(1, 20, alpha=0.1, color='green', label='Phase 1 (Feature Extraction)')
plt.axvspan(20, best_epoch_phase2, alpha=0.1, color='yellow', label='Phase 2 (Good)')
plt.axvspan(best_epoch_phase2, max(epochs_full), alpha=0.1, color='red', label='Overfitting Risk')

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Gap (Train - Validation Accuracy)', fontsize=13)
plt.title('Overfitting Detection: Gap Between Train and Validation Accuracy', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

for i, (epoch, g) in enumerate(zip(epochs_full, gap)):
    if epoch in [1, 5, 10, 15, 20, 25, 30, 35]:
        plt.annotate(f'{g:.2%}', xy=(epoch, g), xytext=(epoch, g + 0.01),
                    fontsize=8, ha='center', color='blue')

plt.tight_layout()
plt.savefig('grafik_bab4_overfitting_gap.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 21. DOWNLOAD MODEL FILES
# ======================
print("\n" + "="*60)
print("DOWNLOAD MODEL FILES")
print("="*60)

files_to_download = [
    'deteksi_makanan_mobilenetv2.keras',
    'deteksi_makanan_mobilenetv2.h5',
    'deteksi_makanan_mobilenetv2.weights.h5',
    'model_info_mobilenetv2.json',
    'confusion_matrix_mobilenetv2.csv',
    'confusion_matrix_mobilenetv2.npy',
    'training_history_mobilenetv2.json',
    'model_summary_mobilenetv2.png'
]

for file_name in files_to_download:
    if os.path.exists(file_name):
        print(f"⬇️  Downloading: {file_name}")
        files.download(file_name)
    else:
        print(f"File not found: {file_name}")

print("\nDownload complete!")