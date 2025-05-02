import sys
import os
from pathlib import Path
import numpy as np
import cv2
import json
import torch
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from custom.feature_extractor import YouTube8MFeatureExtractor
from custom.extract_tfrecords_main import frame_iterator
from networks.pgl_sum.pgl_sum import PGL_SUM

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.current_position = 0.0
        
        # Create layout
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        
        controls_layout.addWidget(self.select_button)
        controls_layout.addWidget(self.play_button)
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
    
    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(file_name)
            self.play_button.setEnabled(True)
            self.current_position = 0.0
            self.update_frame()
    
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(33)  # ~30 fps
            self.play_button.setText("Pause")
    
    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                
                # Update current position
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.current_position = current_frame / total_frames
                
                # Update heatmap position
                if hasattr(self.parent(), 'heatmap_visualizer'):
                    self.parent().heatmap_visualizer.update_current_position(self.current_position)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_position = 0.0

class HeatmapVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(8, 4))  # Make figure wider
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Initialize data storage
        self.original_positions = None
        self.original_heat_values = None
        self.generated_positions = None
        self.generated_heat_values = None
        self.current_position = None
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def load_original_heatmap(self, json_path):
        """Load original heatmap from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.original_positions = np.array([d['position'] for d in data])
            self.original_heat_values = np.array([d['heat'] for d in data])
            self.update_plot()
        except Exception as e:
            print(f"Error loading heatmap JSON: {e}")
    
    def update_generated_heatmap(self, positions, heat_values):
        """Update generated heatmap data"""
        self.generated_positions = positions
        self.generated_heat_values = heat_values
        self.update_plot()
    
    def update_current_position(self, position):
        """Update the current video position indicator"""
        self.current_position = position
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with both original and generated heatmaps"""
        self.ax.clear()
        
        # Plot original heatmap if available
        if self.original_positions is not None and self.original_heat_values is not None:
            self.ax.plot(self.original_positions, self.original_heat_values, 'b-', 
                        label='Original Heatmap', alpha=0.7, linewidth=2)
        
        # Plot generated heatmap if available
        if self.generated_positions is not None and self.generated_heat_values is not None:
            self.ax.plot(self.generated_positions, self.generated_heat_values, 'r-', 
                        label='Generated Heatmap', alpha=0.7, linewidth=2)
        
        # Plot current position indicator
        if self.current_position is not None:
            self.ax.axvline(x=self.current_position, color='g', linestyle='--', 
                          label='Current Position', alpha=0.7)
        
        self.ax.set_xlabel('Time (normalized)')
        self.ax.set_ylabel('Heat Value')
        self.ax.set_title('Video Heatmap')
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Heatmap Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Initialize model
        self.model = None
        self.model_path = None
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create video player
        self.video_player = VideoPlayer()
        main_layout.addWidget(self.video_player)
        
        # Create heatmap visualizer with larger size
        self.heatmap_visualizer = HeatmapVisualizer()
        self.heatmap_visualizer.setMinimumHeight(200)  # Make heatmap taller
        main_layout.addWidget(self.heatmap_visualizer)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Add buttons
        self.load_model_button = QPushButton("Load PGL-SUM Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_heatmap_button = QPushButton("Load Original Heatmap")
        self.load_heatmap_button.clicked.connect(self.load_original_heatmap)
        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.load_heatmap_button)
        button_layout.addWidget(self.process_button)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
    
    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select PGL-SUM Model File", "", "PyTorch Files (*.pkl)"
        )
        if file_name:
            try:
                # Initialize PGL-SUM model
                self.model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8, fusion="add", pos_enc="absolute")
                self.model.load_state_dict(torch.load(file_name))
                self.model.eval()
                self.model_path = file_name
                self.process_button.setEnabled(True)
                print(f"Model loaded successfully from {file_name}")
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def load_original_heatmap(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Heatmap JSON File", "", "JSON Files (*.json)"
        )
        if file_name:
            self.heatmap_visualizer.load_original_heatmap(file_name)
    
    def process_video(self):
        if self.video_player.video_path is None or self.model is None:
            return
        
        try:
            # Extract features and generate heatmap
            model_dir = "../custom/inception_weights/"
            extractor = YouTube8MFeatureExtractor(model_dir=model_dir)
            
            # Get video duration and features
            cap = cv2.VideoCapture(self.video_player.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Extract features from video
            features = []
            for frame in frame_iterator(self.video_player.video_path, every_ms=1000.0, max_num_frames=duration):
                rgb = frame[:, :, ::-1]  # Convert BGR to RGB
                vec = extractor.extract_rgb_frame_features(rgb, apply_pca=True)
                features.append(vec.astype(np.float32))
            
            features = np.stack(features, axis=0)
            cap.release()
            
            # Convert features to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Generate heatmap using PGL-SUM model
            with torch.no_grad():
                score, _ = self.model(features_tensor)
                score = score.squeeze().cpu().numpy()
            
            # Normalize score to [0, 1]
            score = (score - score.min()) / (score.max() - score.min())
            
            # Generate positions (normalized time)
            positions = np.linspace(0, 1, len(score))
            
            # Update heatmap visualization
            self.heatmap_visualizer.update_generated_heatmap(positions, score)
            
        except Exception as e:
            print(f"Error processing video: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 