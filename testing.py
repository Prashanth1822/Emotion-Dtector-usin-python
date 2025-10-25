from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import model_from_json
import threading
import time
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionDetector:
    def __init__(self):
        self.camera = None
        self.model = None
        self.face_cascade = None
        self.is_running = False
        self.current_emotion = "Stopped"
        self.emotion_scores = {}
        self.camera_status = "✗ Error"
        self.model_status = "✗ Error"
        self.detection_status = "Stopped"
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                      4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        # Initialize components
        self.initialize_components()

    def initialize_components(self):
        """Initialize camera and model"""
        self.initialize_model()
        self.initialize_camera()

    def initialize_model(self):
        """Initialize your custom emotion detection model"""
        try:
            # Check if model files exist
            if not os.path.exists("emotiondetector.json"):
                self.model_status = "✗ Error: emotiondetector.json not found"
                logging.error("emotiondetector.json file not found")
                return False
            
            if not os.path.exists("emotiondetector.h5"):
                self.model_status = "✗ Error: emotiondetector.h5 not found"
                logging.error("emotiondetector.h5 file not found")
                return False

            logging.info("Loading emotion detection model...")
            
            # Load model architecture from JSON
            with open("emotiondetector.json", "r") as json_file:
                model_json = json_file.read()
            
            # Create model from JSON
            self.model = model_from_json(model_json)
            
            # Load model weights
            self.model.load_weights("emotiondetector.h5")
            
            # Load face detection cascade
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_file)
            
            if self.face_cascade.empty():
                self.model_status = "✗ Error: Face cascade not loaded"
                logging.error("Failed to load face cascade classifier")
                return False
            
            # Test the model with a dummy prediction
            dummy_input = np.random.random((1, 48, 48, 1)).astype(np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            
            self.model_status = "✓ Loaded"
            logging.info("Custom emotion model loaded successfully")
            return True
            
        except Exception as e:
            self.model_status = f"✗ Error: {str(e)}"
            logging.error(f"Model initialization error: {e}")
            return False

    def initialize_camera(self):
        """Initialize camera with proper error handling"""
        try:
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                # Try different camera indices
                for i in range(1, 3):
                    self.camera = cv2.VideoCapture(i)
                    if self.camera.isOpened():
                        break
            
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                # Test camera
                ret, frame = self.camera.read()
                if ret:
                    self.camera_status = "✓ Connected"
                    logging.info("Camera initialized successfully")
                    return True
                else:
                    self.camera_status = "✗ No frame received"
                    logging.error("Camera connected but no frame received")
            else:
                self.camera_status = "✗ No camera found"
                logging.error("No camera device found")
                
        except Exception as e:
            self.camera_status = f"✗ Error: {str(e)}"
            logging.error(f"Camera initialization error: {e}")
            
        return False

    def extract_features(self, image):
        """Preprocess image for your model"""
        try:
            feature = np.array(image, dtype=np.float32)
            feature = feature.reshape(1, 48, 48, 1)
            return feature / 255.0
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return None

    def detect_emotions_frame(self, frame):
        """Detect emotions in frame using your custom model"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            emotions_data = {}
            processed_frame = frame.copy()
            
            if len(faces) > 0:
                for (p, q, r, s) in faces:
                    try:
                        # Extract face region
                        face_image = gray[q:q+s, p:p+r]
                        face_image = cv2.resize(face_image, (48, 48))
                        
                        # Preprocess and predict
                        img_features = self.extract_features(face_image)
                        if img_features is not None:
                            predictions = self.model.predict(img_features, verbose=0)
                            
                            # Get emotion probabilities
                            emotion_probs = {}
                            for i, emotion in self.labels.items():
                                emotion_probs[emotion] = float(predictions[0][i])
                            
                            # Get dominant emotion
                            dominant_emotion_idx = np.argmax(predictions)
                            dominant_emotion = self.labels[dominant_emotion_idx]
                            confidence = float(np.max(predictions))
                            
                            emotions_data = {
                                'dominant_emotion': dominant_emotion,
                                'probabilities': emotion_probs,
                                'confidence': confidence,
                                'face_coordinates': (p, q, r, s)
                            }
                            
                            # Draw on frame
                            cv2.rectangle(processed_frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
                            cv2.putText(processed_frame, dominant_emotion, (p, q-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Display confidence
                            confidence_text = f"{confidence:.1%}"
                            cv2.putText(processed_frame, confidence_text, (p, q-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                    except Exception as e:
                        logging.error(f"Face processing error: {e}")
                        continue
            else:
                emotions_data = {
                    'dominant_emotion': 'No face detected',
                    'probabilities': {},
                    'confidence': 0.0
                }
            
            return emotions_data, processed_frame
            
        except Exception as e:
            logging.error(f"Emotion detection error: {e}")
            return {}, frame

    def generate_frames(self):
        """Generate video frames with emotion detection"""
        frame_count = 0
        while self.is_running:
            try:
                if self.camera is None or not self.camera.isOpened():
                    logging.error("Camera not available")
                    break
                    
                success, frame = self.camera.read()
                if not success:
                    logging.error("Failed to read frame from camera")
                    break
                
                # Detect emotions if model is loaded
                emotions_data = {}
                if self.model_status == "✓ Loaded" and frame is not None:
                    emotions_data, frame = self.detect_emotions_frame(frame)
                    
                    if emotions_data:
                        self.current_emotion = emotions_data['dominant_emotion']
                        self.emotion_scores = emotions_data['probabilities']
                    else:
                        self.current_emotion = "Analyzing..."
                        self.emotion_scores = {}
                
                # Add status text to frame
                status_text = f"Status: {self.detection_status} | Emotion: {self.current_emotion}"
                cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    logging.error("Failed to encode frame")
                    
                frame_count += 1
                
            except Exception as e:
                logging.error(f"Frame generation error: {e}")
                break

    def start_detection(self):
        """Start emotion detection"""
        try:
            if not self.camera or not self.camera.isOpened():
                if not self.initialize_camera():
                    return False
            
            if self.model_status != "✓ Loaded":
                if not self.initialize_model():
                    return False
            
            self.is_running = True
            self.detection_status = "Running"
            self.current_emotion = "Analyzing..."
            logging.info("Emotion detection started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting detection: {e}")
            return False

    def stop_detection(self):
        """Stop emotion detection"""
        self.is_running = False
        self.detection_status = "Stopped"
        self.current_emotion = "Stopped"
        self.emotion_scores = {}
        logging.info("Emotion detection stopped")

    def release_camera(self):
        """Release camera resources"""
        if self.camera and self.camera.isOpened():
            self.camera.release()
        self.camera_status = "✗ Disconnected"

# Global detector instance
detector = EmotionDetector()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        return Response(detector.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"Video feed error: {e}")
        return "Video feed error", 500

@app.route('/start', methods=['POST'])
def start_detection():
    """Start detection endpoint"""
    try:
        if detector.start_detection():
            return jsonify({
                "status": "success", 
                "message": "Detection started successfully"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to start detection. Check camera and model."
            }), 400
    except Exception as e:
        logging.error(f"Start detection error: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop detection endpoint"""
    try:
        detector.stop_detection()
        return jsonify({
            "status": "success", 
            "message": "Detection stopped successfully"
        })
    except Exception as e:
        logging.error(f"Stop detection error: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/status')
def get_status():
    """Get system status endpoint"""
    try:
        return jsonify({
            "current_emotion": detector.current_emotion,
            "camera_status": detector.camera_status,
            "model_status": detector.model_status,
            "detection_status": detector.detection_status,
            "emotion_scores": detector.emotion_scores
        })
    except Exception as e:
        logging.error(f"Status endpoint error: {e}")
        return jsonify({
            "current_emotion": "Error",
            "camera_status": "✗ Error",
            "model_status": "✗ Error", 
            "detection_status": "Error",
            "emotion_scores": {}
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Real-Time Emotion Detector Server...")
    print("📁 Make sure these files are in the same directory:")
    print("   - emotiondetector.json")
    print("   - emotiondetector.h5")
    print("   - templates/index.html")
    print("\n🌐 Server will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Start the Flask application
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        threaded=True,
        use_reloader=False
    )