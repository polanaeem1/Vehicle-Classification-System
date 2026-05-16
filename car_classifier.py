#!/usr/bin/env python3
"""
Car Classification AI Tool (Computer Vision)
Uses YOLOv8 for vehicle detection and classification with parking recommendations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from io import BytesIO

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from PIL import Image
    import requests
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class CarClassifier:
    """Main class for car classification using computer vision."""
    
    # COCO dataset vehicle classes
    COCO_VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    # Parking spot type mappings
    SPOT_MAPPINGS = {
        "sedan": "standard",
        "suv": "standard",
        "truck": "oversized",
        "van": "oversized",
        "motorcycle": "motorcycle",
        "compact": "compact",
        "hatchback": "compact",
        "bus": "oversized",
        "car": "standard"
    }
    
    # Confidence thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    
    # Image quality thresholds
    BLUR_THRESHOLD = 100.0  # Laplacian variance
    LOW_BRIGHTNESS_THRESHOLD = 50
    HIGH_BRIGHTNESS_THRESHOLD = 200
    
    def __init__(self, detector_model: str = 'yolov8n.pt', classifier_model: str = 'best.pt', verbose: bool = False):
        """
        Initialize the CarClassifier.
        
        Args:
            detector_model: YOLOv8 model to use for DETECTION (yolov8n.pt)
            classifier_model: Custom YOLOv8 model to use for CLASSIFICATION (best.pt)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._log("Initializing CarClassifier...")
        
        # 1. Load Detector (Finds the car)
        try:
            self.detector = YOLO(detector_model)
            self._log(f"Detector model '{detector_model}' loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load detector model: {e}")

        # 2. Load Custom Classifier (Identifies the type)
        self.classifier_model = None
        if os.path.exists(classifier_model):
            try:
                self.classifier_model = YOLO(classifier_model)
                self._log(f"Custom classifier '{classifier_model}' loaded successfully")
            except Exception as e:
                self._log(f"Warning: Failed to load custom classifier: {e}. Falling back to heuristics.")
        else:
             self._log(f"Warning: Custom classifier '{classifier_model}' not found. Falling back to heuristics.")
    
    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def load_image(self, image_source: str) -> np.ndarray:
        """
        Load image from file path or URL.
        
        Args:
            image_source: File path or URL to image
            
        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        self._log(f"Loading image from: {image_source}")
        
        # Check if it's a URL
        if image_source.startswith(('http://', 'https://')):
            return self._load_from_url(image_source)
        else:
            return self._load_from_file(image_source)
    
    def _load_from_url(self, url: str) -> np.ndarray:
        """Load image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from URL")
            
            self._log(f"Successfully loaded image from URL ({image.shape})")
            return image
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to load image from URL: {e}")
    
    def _load_from_file(self, file_path: str) -> np.ndarray:
        """Load image from file path."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        image = cv2.imread(str(path))
        
        if image is None:
            raise ValueError(f"Failed to load image. Unsupported format or corrupted file.")
        
        self._log(f"Successfully loaded image from file ({image.shape})")
        return image
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess image quality (brightness, blur, etc.).
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.BLUR_THRESHOLD
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        if brightness < self.LOW_BRIGHTNESS_THRESHOLD:
            lighting = "dark"
        elif brightness > self.HIGH_BRIGHTNESS_THRESHOLD:
            lighting = "bright"
        else:
            lighting = "good"
        
        # Determine overall quality
        if is_blurry or lighting in ["dark", "bright"]:
            quality = "poor"
        elif laplacian_var < self.BLUR_THRESHOLD * 1.5:
            quality = "fair"
        else:
            quality = "good"
        
        return {
            "quality": quality,
            "lighting": lighting,
            "blur_score": float(laplacian_var),
            "is_blurry": is_blurry,
            "brightness": float(brightness)
        }
    
    def classify(self, image_source: str) -> Dict[str, Any]:
        """
        Classify a vehicle image and provide parking recommendations.
        
        Args:
            image_source: Path to image file or URL
            
        Returns:
            Dictionary with classification results
        """
        self._log("Starting classification process")
        
        try:
            # Load image
            image = self.load_image(image_source)
            
            # Assess image quality
            quality_metrics = self.assess_image_quality(image)
            self._log(f"Image quality: {quality_metrics['quality']}, lighting: {quality_metrics['lighting']}")
            
            # 1. Run YOLO detection (Stage 1: Find the car)
            self._log("Stage 1: Running Object Detection...")
            results = self.detector(image, verbose=False)
            
            # Extract vehicle detections
            detections = self._extract_vehicle_detections(results[0], image.shape)
            
            if not detections:
                return self._create_error_response(
                    "No vehicle detected in image",
                    quality_metrics
                )
            
            # Get the most prominent vehicle (largest bounding box)
            main_vehicle = max(detections, key=lambda x: x['area'])
            
            # 2. Run Classification (Stage 2: Identify Type)
            vehicle_type, class_conf = self._predict_vehicle_type(main_vehicle, image)
            
            # Check for multiple vehicles
            warnings = []
            if len(detections) > 1:
                warnings.append(f"Multiple vehicles detected ({len(detections)}). Classifying the largest one.")
            
            # Add quality warnings
            if quality_metrics['quality'] == 'poor':
                warnings.append("Poor image quality detected. Results may be less accurate.")
            if quality_metrics['lighting'] == 'dark':
                warnings.append("Low-light conditions detected. Consider re-uploading with better lighting.")
            if quality_metrics['is_blurry']:
                warnings.append("Image appears blurry. Consider uploading a clearer image.")
            
            # Check confidence
            if class_conf < self.LOW_CONFIDENCE_THRESHOLD:
                warnings.append(f"Low confidence ({class_conf:.2f}). Results may not be reliable.")
            
            # Determine visibility
            visibility = self._assess_visibility(main_vehicle, image.shape)
            
            # Build result
            result = {
                'status': 'success',
                'car_type': vehicle_type,
                'confidence': float(class_conf),
                'base_class': main_vehicle['class_name'],
                'image_quality': quality_metrics['quality'],
                'visibility': visibility,
                'lighting_conditions': quality_metrics['lighting'],
                'warnings': warnings,
                'detection_details': {
                    'bbox': main_vehicle['bbox'],
                    'area_percentage': main_vehicle['area_percentage']
                }
            }
            
            # Add parking recommendations
            result = self._add_parking_recommendations(result)
            
            return result
            
        except Exception as e:
            self._log(f"Classification error: {e}")
            return self._create_error_response(str(e))
    
    def _extract_vehicle_detections(self, result, image_shape: tuple) -> List[Dict[str, Any]]:
        """Extract vehicle detections from YOLO results."""
        detections = []
        
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        
        height, width = image_shape[:2]
        image_area = height * width
        
        for box in boxes:
            class_id = int(box.cls[0])
            
            # Check if it's a vehicle class
            if class_id in self.COCO_VEHICLE_CLASSES:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate area
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                area = bbox_width * bbox_height
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.COCO_VEHICLE_CLASSES[class_id],
                    'confidence': float(box.conf[0]),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'width': float(bbox_width),
                    'height': float(bbox_height),
                    'area': float(area),
                    'area_percentage': float(area / image_area * 100),
                    'aspect_ratio': float(bbox_width / bbox_height) if bbox_height > 0 else 0
                })
        
        return detections

    def _predict_vehicle_type(self, detection: Dict[str, Any], image: np.ndarray) -> Tuple[str, float]:
        """
        Identify vehicle type using Custom Classifier (Stage 2) or Heuristics (Fallback).
        Returns: (vehicle_type, confidence)
        """
        # Crop the vehicle image
        x1, y1, x2, y2 = detection['bbox']
        # Add small padding if possible
        h, w, _ = image.shape
        pad = 10
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        car_crop = image[y1:y2, x1:x2]
        
        if car_crop.size == 0:
            return detection['class_name'], detection['confidence']

        # OPTION A: Use Custom Trained Model (Best)
        if self.classifier_model:
            self._log("Stage 2: Running Custom Classifier...")
            try:
                # Run inference on crop
                results = self.classifier_model(car_crop, verbose=False)
                # Get top prediction
                probs = results[0].probs
                top1_idx = probs.top1
                confidence = probs.top1conf.item()
                vehicle_type = results[0].names[top1_idx]
                
                self._log(f"Custom Model Prediction: {vehicle_type} ({confidence:.2f})")
                return vehicle_type, confidence
            except Exception as e:
                self._log(f"Custom classifier failed: {e}. Falling back to heuristics.")

        # OPTION B: Heuristic Fallback (Legacy)
        self._log("Using Heuristic Fallback...")
        return self._heuristic_classification(detection), detection['confidence']

    def _heuristic_classification(self, detection: Dict[str, Any]) -> str:
        """Legacy heuristic classification based on box shape."""
        base_class = detection['class_name']
        aspect_ratio = detection['aspect_ratio']
        area_percentage = detection['area_percentage']
        
        if base_class == 'motorcycle': return 'motorcycle'
        if base_class == 'bus': return 'bus'
        if base_class == 'truck': return 'truck'
        
        if base_class == 'car':
            if area_percentage < 15 or aspect_ratio < 1.4: return 'compact'
            elif aspect_ratio < 1.7 and area_percentage > 20: return 'suv'
            elif aspect_ratio < 1.5 and area_percentage > 25: return 'van'
            else: return 'sedan'
            
        return base_class
    
    def _assess_visibility(self, detection: Dict[str, Any], image_shape: tuple) -> str:
        """Assess how much of the vehicle is visible."""
        x1, y1, x2, y2 = detection['bbox']
        height, width = image_shape[:2]
        
        # Check if bounding box is cut off by image edges
        edge_threshold = 5  # pixels
        
        is_cut_left = x1 < edge_threshold
        is_cut_right = x2 > width - edge_threshold
        is_cut_top = y1 < edge_threshold
        is_cut_bottom = y2 > height - edge_threshold
        
        is_cut = is_cut_left or is_cut_right or is_cut_top or is_cut_bottom
        
        # Check area percentage
        area_percentage = detection['area_percentage']
        
        if is_cut or area_percentage < 10:
            return 'partial'
        elif area_percentage > 40:
            return 'full'
        else:
            return 'good'
    
    def _add_parking_recommendations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add parking spot recommendations based on classification."""
        if result.get('status') == 'error':
            return result
        
        car_type = result['car_type']
        confidence = result['confidence']
        
        # Determine spot type
        spot_type = self.SPOT_MAPPINGS.get(car_type, 'standard')
        
        # Generate optimization tips
        optimization_tips = self._generate_optimization_tips(
            car_type, spot_type, confidence
        )
        
        result['parking_recommendation'] = {
            'spot_type': spot_type,
            'priority': self._get_priority_level(car_type),
            'optimization_tips': optimization_tips
        }
        
        return result
    
    def _generate_optimization_tips(
        self, car_type: str, spot_type: str, confidence: float
    ) -> List[str]:
        """Generate parking lot optimization tips."""
        tips = []
        
        # Confidence-based tips
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            tips.append("Manual verification recommended due to low confidence")
        
        # Vehicle-specific tips
        if car_type == 'motorcycle':
            tips.append("Allocate to designated motorcycle zone for space efficiency")
            tips.append("Group motorcycles together to maximize parking capacity")
        elif car_type in ['truck', 'van', 'bus']:
            tips.append("Assign to oversized spots near entrance/exit for easier maneuvering")
            tips.append("Avoid compact zones to prevent overflow")
        elif car_type in ['compact', 'hatchback']:
            tips.append("Ideal for compact spots to maximize space utilization")
        elif car_type == 'suv':
            tips.append("Suitable for standard spots with adequate clearance")
            tips.append("Consider grouping SUVs in dedicated zones")
        elif car_type == 'sedan':
            tips.append("Optimal for standard parking spaces")
            tips.append("Most flexible for various parking zones")
        
        # General optimization
        tips.append(f"Group similar vehicle types ({car_type}) together for better flow")
        
        return tips
    
    def _get_priority_level(self, car_type: str) -> str:
        """Determine parking priority level."""
        if car_type == 'motorcycle':
            return "medium"  # Specific zone needed
        elif car_type in ['truck', 'bus', 'van']:
            return "high"  # Need oversized spots
        else:
            return "standard"
    
    def _create_error_response(
        self, error_message: str, quality_metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        response = {
            'status': 'error',
            'error': error_message,
            'car_type': 'unknown',
            'confidence': 0.0,
            'warnings': ['Classification failed - please check image and try again']
        }
        
        if quality_metrics:
            response['image_quality'] = quality_metrics['quality']
            response['lighting_conditions'] = quality_metrics['lighting']
        
        return response


def main():
    """Command-line interface for the car classifier."""
    parser = argparse.ArgumentParser(
        description='Car Classification AI Tool using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python car_classifier.py car.jpg
  python car_classifier.py car.jpg --model best.pt
        """
    )
    
    parser.add_argument('image', help='Path to image file or URL')
    parser.add_argument('--output', '-o', help='Save results to JSON file', metavar='FILE')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--model', '-m', default='best.pt', help='Custom classifier model (e.g. best.pt). Default: best.pt')
    
    args = parser.parse_args()
    
    try:
        # Initialize classifier with custom model
        print("Initializing car classifier...")
        # Note: we use standard yolov8n.pt for detection (args are for classifier)
        classifier = CarClassifier(detector_model='yolov8n.pt', classifier_model=args.model, verbose=args.verbose)
        
        # Classify image
        print(f"Classifying image: {args.image}")
        result = classifier.classify(args.image)
        
        # Format output
        output_json = json.dumps(result, indent=2)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            print(f"\nResults saved to: {args.output}")
        
        # Always print to console
        print("\n" + "="*60)
        print("CAR CLASSIFICATION RESULTS")
        print("="*60)
        print(output_json)
        print("="*60 + "\n")
        
        # Exit with appropriate code
        sys.exit(0 if result.get('status') != 'error' else 1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
