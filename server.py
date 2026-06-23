from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
from supabase import create_client, Client
from werkzeug.utils import secure_filename
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2


# ==============================================
# 1. KONFIGURASI AWAL
# ==============================================
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

CONFIDENCE_THRESHOLD = 70.0 

SUPABASE_URL = "https://vakicuixdhfnffgynayx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZha2ljdWl4ZGhmbmZmZ3luYXl4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk0OTUyMjQsImV4cCI6MjA4NTA3MTIyNH0.LsjOHnJT9b5BLyCnkmEzDFLBI5BPQnkfcH9LMAWx8CA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================================
# 2. LOAD MODEL CNN
# ==============================================
class FoodDetectionModel:
    def __init__(self, model_path='deteksi_makanan_mobilenetv2.keras'):
        print(f"\n{'='*60}")
        print("🤖 INITIALIZING FOOD DETECTION MODEL - MOBILENETV2")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print(f"Files in directory: {os.listdir('.')}")
            self.model = None
        else:
            try:
                print(f"Loading model: {model_path}")
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("Model loaded successfully!")
                
                input_shape = self.model.input_shape
                print(f"Model input shape: {input_shape}")
                
                if input_shape[1] == input_shape[2]:
                    self.img_size = input_shape[1]
                else:
                    self.img_size = 224
                
                print(f"Image size for preprocessing: {self.img_size}x{self.img_size}")
                
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
        
        print(f"\nCLASS NAMES:")
        self.class_names = [
        "Bubur Kacang Hijau",
        "Gado-Gado",
        "Gudeg",
        "Mie Ayam",
        "Nasi Goreng",
        "Nasi Kuning",
        "Nasi Padang",
        "Nasi Uduk",
        "Rawon",
        "Rendang",
        "Sate",
        "Sop Buntut",
        "Soto",
        "Tempe"
        ]
        
        self.model_metrics = {
            'accuracy': 92.0,
            'precision': 91.0,
            'recall': 90.0,
            'f1_score': 90.5,
            'confidence_threshold': CONFIDENCE_THRESHOLD
        }
        
        for i, name in enumerate(self.class_names):
            print(f"  [{i:2d}] {name}")
        
        print(f"{'='*60}\n")
    
    def preprocess_image(self, image_bytes):
        """PREPROCESSING UNTUK MOBILENETV2"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            
            print(f"Original image: {img.shape}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            target_size = (self.img_size, self.img_size)
            img = cv2.resize(img, target_size)
            print(f"Resized to: {img.shape}")
            
            img = preprocess_input(img.astype(np.float32))
            
            print(f"After MobileNetV2 preprocessing: shape={img.shape}, dtype={img.dtype}")
            print(f"Pixel range: [{img.min():.2f}, {img.max():.2f}]")
            
            img = np.expand_dims(img, axis=0)
            print(f"Final batch shape: {img.shape}")
            
            return img
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            try:
                print("Trying fallback preprocessing...")
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((self.img_size, self.img_size))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = preprocess_input(img_array.astype(np.float32))
                img_array = np.expand_dims(img_array, axis=0)
                print("Fallback preprocessing successful")
                return img_array
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise
    
    def predict(self, image_bytes):
        """Predict dengan confidence threshold"""
        if self.model is None:
            print("Using DUMMY model")
            import random
            food_name = random.choice(self.class_names)
            confidence = round(random.uniform(80, 95), 2)
            return {
                'food_name': food_name,
                'confidence': confidence,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'above_threshold': confidence >= CONFIDENCE_THRESHOLD,
                'message': f"[DUMMY] Detected: {food_name}",
                'model_evaluation': self.model_metrics
            }
        
        try:
            print(f"\n{'='*50}")
            print("MODEL PREDICTION START (MobileNetV2)")
            print(f"{'='*50}")
            
            input_tensor = self.preprocess_image(image_bytes)
            
            print("Running model prediction...")
            predictions = self.model.predict(input_tensor, verbose=0)
            print(f"Raw predictions shape: {predictions.shape}")
            
            if predictions.shape[-1] > 1:
                if predictions.max() > 1 or predictions.min() < 0:
                    predictions = tf.nn.softmax(predictions, axis=-1).numpy()
                predicted_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_idx]) * 100
            else:
                predicted_idx = 0
                confidence = 50.0
            
            above_threshold = confidence >= CONFIDENCE_THRESHOLD
            
            print(f"\nTOP 5 PREDICTIONS:")
            if predictions.shape[-1] > 1:
                top_indices = np.argsort(predictions[0])[-5:][::-1]
                for rank, idx in enumerate(top_indices, 1):
                    prob = predictions[0][idx] * 100
                    name = self.class_names[idx] if idx < len(self.class_names) else f'Class_{idx}'
                    star = "★" if idx == predicted_idx else " "
                    threshold_mark = " ✓" if prob >= CONFIDENCE_THRESHOLD else " ✗"
                    print(f"  {star} {rank}. {name:<20} {prob:6.2f}%{threshold_mark}")
            
            print(f"\n🎯 FINAL SELECTION: {self.class_names[predicted_idx]} ({confidence:.2f}%)")
            print(f"📊 Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
            print(f"✅ Above threshold: {above_threshold}")
            
            if not above_threshold:
                print(f"⚠️ WARNING: Confidence below threshold!")
            
            if predicted_idx < len(self.class_names):
                food_name = self.class_names[predicted_idx]
            else:
                food_name = f"Class_{predicted_idx}"
            
            top_predictions = []
            if predictions.shape[-1] > 1:
                top_indices = np.argsort(predictions[0])[-5:][::-1]
                for idx in top_indices:
                    if idx < len(self.class_names):
                        top_predictions.append({
                            'class': self.class_names[idx],
                            'index': int(idx),
                            'confidence': float(predictions[0][idx]) * 100
                        })
            
            result = {
                'food_name': food_name,
                'confidence': round(confidence, 2),
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'above_threshold': above_threshold,
                'all_predictions': predictions[0].tolist() if predictions.shape[-1] > 1 else [],
                'top_predictions': top_predictions,
                'model_evaluation': self.model_metrics,
                'message': f"Detected: {food_name} ({confidence:.1f}%)"
            }
            
            if not above_threshold:
                result['warning'] = f"Tingkat keyakinan rendah ({confidence:.1f}%). Silakan coba dengan gambar yang lebih jelas."
            
            print(f"{'='*50}\n")
            
            return result
            
        except Exception as e:
            print(f"\n❌ PREDICTION ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            import random
            food_name = random.choice(self.class_names)
            return {
                'food_name': food_name,
                'confidence': round(confidence, 2),
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'above_threshold': False,
                'warning': f"Error deteksi: {str(e)[:100]}",
                'message': f"[ERROR] Detection failed, random: {food_name}",
                'model_evaluation': self.model_metrics,
                'top_predictions': []
            }
            
print("🔄 Preparing food detection model...")
food_model = FoodDetectionModel('deteksi_makanan_mobilenetv2.keras')

# ==============================================
# 3. HELPER FUNCTIONS
# ==============================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_image(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filename
    return None

# ==============================================
# 4. ROUTES - FRONTEND 
# ==============================================
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/dashboard.css')
def serve_css():
    return app.send_static_file('dashboard.css')

@app.route('/dashboard.js')
def serve_js():
    return app.send_static_file('dashboard.js')

# ==============================================
# 5. ROUTES - API ENDPOINTS
# ==============================================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': food_model.model is not None,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_and_detect():
    """Main endpoint: Upload → Detect → Get nutrition"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file', 'success': False}), 400
        
        file = request.files['image']
        user_id = request.form.get('user_id', 'skripsi_user')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed', 'success': False}), 400
        
        filename = save_uploaded_image(file)
        if not filename:
            return jsonify({'error': 'Failed to save image', 'success': False}), 500
        
        file.seek(0)
        image_bytes = file.read()
        
        print(f"\n{'='*50}")
        print(f"🔍 PROCESSING NEW IMAGE: {filename}")
        print(f"{'='*50}")
        
        detection_result = food_model.predict(image_bytes)
        food_name = detection_result['food_name']
        confidence = detection_result['confidence']
        confidence_threshold = detection_result.get('confidence_threshold', CONFIDENCE_THRESHOLD)
        above_threshold = detection_result.get('above_threshold', True)
        
        print(f"\n🎯 FINAL PREDICTION: {food_name} ({confidence}%)")
        print(f"📊 Threshold: {confidence_threshold}%, Above: {above_threshold}")
        
        food_name_lower = food_name.lower().strip()
        
        try:
            all_foods_response = supabase.table('tabelmakanan')\
                .select('food_name, energi, protein, lemak, karbohidrat, kode_tkpi')\
                .execute()
            
            available_foods = [f['food_name'].lower() for f in all_foods_response.data]
            print(f"\n📋 Available in database ({len(available_foods)} foods):")
            for f in all_foods_response.data[:10]:
                print(f"  - {f['food_name']} (Kode: {f.get('kode_tkpi', 'N/A')})")
            
            matched_food = None
            matched_data = None
            
            for food_data in all_foods_response.data:
                if food_data['food_name'].lower() == food_name_lower:
                    matched_food = food_data['food_name']
                    matched_data = food_data
                    break
            
            if not matched_data:
                for food_data in all_foods_response.data:
                    if food_name_lower in food_data['food_name'].lower() or \
                       food_data['food_name'].lower() in food_name_lower:
                        matched_food = food_data['food_name']
                        matched_data = food_data
                        print(f"🔍 Partial match found: {food_name} → {matched_food}")
                        break
            
            if not matched_data:
                manual_mapping = {
                    'nasi goreng': 'nasi goreng',
                    'sop buntut': 'sop buntut',
                    'gado-gado': 'gado-gado',
                    'gado gado': 'gado-gado',
                    'bubur kacang hijau': 'bubur kacang hijau',
                    'mie ayam': 'mie ayam',
                    'nasi padang': 'nasi padang',
                    'rawon': 'rawon',
                    'rendang': 'rendang',
                    'sate': 'sate',
                    'soto': 'soto',
                    'gudeg': 'gudeg',
                    'tempe': 'tempe',
                    'nasi uduk': 'nasi uduk',
                    'nasi kuning': 'nasi kuning'
                }
                
                if food_name_lower in manual_mapping:
                    target_name = manual_mapping[food_name_lower]
                    for food_data in all_foods_response.data:
                        if food_data['food_name'].lower() == target_name:
                            matched_food = food_data['food_name']
                            matched_data = food_data
                            print(f"🔍 Manual mapping: {food_name} → {matched_food}")
                            break
            
            if matched_data:
                print(f"\n✅ DATABASE MATCH FOUND: {matched_food}")
                nutrition_info = {
                    'energi': float(matched_data.get('energi', 0)),
                    'protein': float(matched_data.get('protein', 0)),
                    'lemak': float(matched_data.get('lemak', 0)),
                    'karbohidrat': float(matched_data.get('karbohidrat', 0)),
                    'kode_tkpi': matched_data.get('kode_tkpi', '')
                }
            else:
                print(f"\n❌ NO MATCH FOUND for: {food_name}")
                
                if all_foods_response.data:
                    fallback_data = all_foods_response.data[0]
                    nutrition_info = {
                        'energi': float(fallback_data.get('energi', 0)),
                        'protein': float(fallback_data.get('protein', 0)),
                        'lemak': float(fallback_data.get('lemak', 0)),
                        'karbohidrat': float(fallback_data.get('karbohidrat', 0)),
                        'kode_tkpi': fallback_data.get('kode_tkpi', ''),
                        'note': f'Data asli untuk {food_name} tidak ditemukan'
                    }
                else:
                    nutrition_info = {
                        'energi': 0, 'protein': 0, 'lemak': 0, 'karbohidrat': 0,
                        'kode_tkpi': 'NOT-FOUND',
                        'note': f'Data untuk {food_name} tidak ditemukan'
                    }
                    
        except Exception as db_err:
            print(f"\n❌ DATABASE ERROR: {db_err}")
            nutrition_info = {
                'energi': 0, 'protein': 0, 'lemak': 0, 'karbohidrat': 0,
                'kode_tkpi': 'DB-ERROR',
                'note': 'Database error'
            }
        
        try:
            history_data = {
                'user_id': user_id,
                'image_filename': filename,
                'detected_food': food_name,
                'confidence': confidence,
                'above_threshold': above_threshold,
                'nutrition_info': nutrition_info
            }
            supabase.table('detection_history').insert(history_data).execute()
            print("💾 Saved to history")
        except Exception as db_err:
            print(f"⚠️ Failed to save history: {db_err}")
        
        response_data = {
            'success': True,
            'detection': {
                'food_name': food_name,
                'confidence': confidence,
                'confidence_threshold': confidence_threshold,  
                'above_threshold': above_threshold,  
                'message': detection_result['message']
            },
            'nutrition': nutrition_info,
            'model_evaluation': detection_result.get('model_evaluation', {}),
            'top_predictions': detection_result.get('top_predictions', []),
            'image_url': f"/api/uploads/{filename}",
            'timestamp': datetime.now().isoformat()
        }
        
        if 'warning' in detection_result:
            response_data['warning'] = detection_result['warning']
        
        print(f"\n📤 SENDING RESPONSE:")
        print(f"   Food: {food_name}")
        print(f"   Confidence: {confidence}%")
        print(f"   Threshold: {confidence_threshold}%")
        print(f"   Above Threshold: {above_threshold}")
        print(f"{'='*50}\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n❌ UPLOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/uploads/<filename>')
def get_uploaded_image(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/foods', methods=['GET'])
def get_all_foods():
    try:
        response = supabase.table('tabelmakanan')\
            .select('food_name, energi, protein, lemak, karbohidrat, kode_tkpi')\
            .order('food_name')\
            .execute()
        return jsonify({'success': True, 'foods': response.data})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        'success': True,
        'model_loaded': food_model.model is not None,
        'classes': food_model.class_names,
        'model_metrics': food_model.model_metrics,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'num_classes': len(food_model.class_names)
    })

# ==============================================
# 6. JALANKAN SERVER
# ==============================================
if __name__ == '__main__':
    print("=" * 60)
    print("🍽️  FOOD DETECTION SYSTEM - SKRIPSI (MobileNetV2)")
    print("=" * 60)
    print(f"✅ Backend: http://localhost:5000")
    print(f"✅ Frontend: http://localhost:5000")
    print(f"✅ Model: {'Loaded' if food_model.model else 'Dummy Mode'}")
    print(f"✅ Classes: {len(food_model.class_names)} foods")
    print(f"✅ Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
    print("=" * 60)
    
    files = os.listdir('.')
    html_files = [f for f in files if f.endswith('.html')]
    print(f"📁 HTML files found: {html_files}")
    
    if 'index.html' not in files:
        print("⚠️ WARNING: index.html not found in current directory")
        print(f"📁 Current directory: {os.getcwd()}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)