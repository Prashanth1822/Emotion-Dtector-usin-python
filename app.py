import cv2
import numpy as np
from keras.models import model_from_json
import time
import os
from datetime import datetime

class EmotionDetector:
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                      4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.is_running = False
        
    def load_model(self):
        """Load the emotion detection model"""
        try:
            print("🔧 Loading emotion detection model...")
            json_file = open("emotiondetector.json", "r")
            model_json = json_file.read()
            json_file.close()
            
            self.model = model_from_json(model_json)
            self.model.load_weights("emotiondetector.h5")
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_face_detector(self):
        """Load face detection classifier"""
        try:
            print("🔧 Loading face detection classifier...")
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_file)
            
            if self.face_cascade.empty():
                raise Exception("Failed to load cascade classifier")
                
            print("✅ Face detector loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading face detector: {e}")
            return False
    
    def extract_features(self, image):
        """Preprocess image for the model"""
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def create_emotion_chart(self, emotion_probs, width=200, height=150):
        """Create a bar chart visualization of emotion probabilities"""
        chart = np.zeros((height, width, 3), dtype=np.uint8)
        bar_width = width // len(emotion_probs)
        
        emotions = list(emotion_probs.keys())
        probabilities = list(emotion_probs.values())
        
        for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
            bar_height = int(prob * (height - 20))
            color = self.get_emotion_color(emotion)
            
            # Draw bar
            x1 = i * bar_width
            x2 = (i + 1) * bar_width - 2
            y1 = height - bar_height
            y2 = height - 5
            
            cv2.rectangle(chart, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(chart, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # Draw emotion name (abbreviated)
            abbrev = emotion[:3].upper()
            text_size = cv2.getTextSize(abbrev, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = x1 + (bar_width - text_size[0]) // 2
            text_y = height - 8
            
            cv2.putText(chart, abbrev, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return chart
    
    def get_emotion_color(self, emotion):
        """Get color for each emotion"""
        colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255)  # Orange
        }
        return colors.get(emotion, (255, 255, 255))
    
    def run_detection(self):
        """Main function to run emotion detection"""
        if not self.load_model() or not self.load_face_detector():
            return
        
        print("🔧 Initializing webcam...")
        webcam = cv2.VideoCapture(0)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        webcam.set(cv2.CAP_PROP_FPS, 30)
        
        if not webcam.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("✅ Webcam initialized successfully!")
        print("\n🎮 Controls:")
        print("  Q - Quit")
        print("  R - Reset statistics")
        print("  S - Save current frame")
        print("  C - Toggle emotion chart")
        print("  D - Toggle debug info")
        print("\n🚀 Starting emotion detection...")
        
        # Statistics
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        emotion_count = {emotion: 0 for emotion in self.labels.values()}
        total_faces = 0
        
        # Settings
        show_chart = True
        show_debug = True
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = webcam.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            current_emotions = []
            
            for (x, y, w, h) in faces:
                try:
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi = cv2.resize(face_roi, (48, 48))
                    
                    img_features = self.extract_features(face_roi)
                    predictions = self.model.predict(img_features, verbose=0)
                    
                    emotion_probs = {}
                    for i, emotion in self.labels.items():
                        emotion_probs[emotion] = float(predictions[0][i])
                    
                    dominant_idx = np.argmax(predictions)
                    dominant_emotion = self.labels[dominant_idx]
                    confidence = np.max(predictions)
                    
                    emotion_count[dominant_emotion] += 1
                    total_faces += 1
                    current_emotions.append((dominant_emotion, confidence))
                    
                    # Draw face rectangle
                    color = self.get_emotion_color(dominant_emotion)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw emotion label
                    label = f"{dominant_emotion} ({confidence:.1%})"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                                 (x + text_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw emotion chart for this face if enabled
                    if show_chart and len(faces) <= 2:  # Limit charts to avoid clutter
                        chart = self.create_emotion_chart(emotion_probs)
                        chart_x = x + w + 10
                        chart_y = y
                        
                        if chart_x + chart.shape[1] < frame.shape[1]:
                            frame[chart_y:chart_y + chart.shape[0], 
                                  chart_x:chart_x + chart.shape[1]] = chart
                    
                except Exception as e:
                    continue
            
            # Display FPS and statistics
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            if show_debug:
                y_offset = 30
                info_lines = [
                    f"FPS: {fps}",
                    f"Faces: {len(faces)}",
                    f"Total Detections: {total_faces}"
                ]
                
                for line in info_lines:
                    cv2.putText(frame, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Show current emotions
                if current_emotions:
                    emotion_text = "Current: " + ", ".join([f"{e}({c:.0%})" for e, c in current_emotions])
                    cv2.putText(frame, emotion_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 20
            
            # Display controls help
            controls = "Q:Quit R:Reset S:Save C:Chart D:Debug"
            cv2.putText(frame, controls, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Real-Time Emotion Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                emotion_count = {emotion: 0 for emotion in self.labels.values()}
                total_faces = 0
                print("📊 Statistics reset")
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved as {filename}")
            elif key == ord('c'):
                show_chart = not show_chart
                print(f"📊 Emotion chart: {'ON' if show_chart else 'OFF'}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"🔧 Debug info: {'ON' if show_debug else 'OFF'}")
        
        # Cleanup
        webcam.release()
        cv2.destroyAllWindows()
        self.is_running = False
        
        # Print session summary
        self.print_session_summary(emotion_count, total_faces)
    
    def print_session_summary(self, emotion_count, total_faces):
        """Print session statistics"""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        print(f"Total faces analyzed: {total_faces}")
        
        if total_faces > 0:
            print("\nEmotion Distribution:")
            for emotion, count in sorted(emotion_count.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / total_faces) * 100
                    bar = "█" * int(percentage / 5)  # Each █ represents 5%
                    print(f"  {emotion:8} : {bar} {count:3d} ({percentage:5.1f}%)")
            
            most_common = max(emotion_count.items(), key=lambda x: x[1])
            print(f"\n🎭 Most common emotion: {most_common[0]} ({most_common[1]} times)")
        
        print("👋 Session ended successfully!")

# Run the emotion detector
if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run_detection()