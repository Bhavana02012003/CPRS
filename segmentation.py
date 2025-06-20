import os
import sys
import cv2
import json
import numpy as np
from ultralytics import YOLO
import logging

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Process an uploaded image using YOLO models for segmentation
    Returns a dictionary with results and metadata
    """
    try:
        # Configuration
        GENERAL_MODEL_PATH = "best.pt"
        PORTS_MODEL_PATH = "port_best.pt"
        OUTPUT_DIR = "static/segmented_outputs"
        CONF_THRESHOLD = 0.2
        COORDINATES_FILE = "static/segmented_outputs/coordinates.json"
        MARGIN = 10
        MIN_DIM = 128
        
        # Check if models exist
        if not os.path.exists(GENERAL_MODEL_PATH):
            return {
                'success': False, 
                'error': f'General model not found: {GENERAL_MODEL_PATH}. Please ensure the model file is in the root directory.'
            }
        
        if not os.path.exists(PORTS_MODEL_PATH):
            return {
                'success': False, 
                'error': f'Ports model not found: {PORTS_MODEL_PATH}. Please ensure the model file is in the root directory.'
            }
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {'success': False, 'error': f'Could not load image: {image_path}'}
        
        # Define class mapping
        TARGET_CLASS_MAP = {
            'Cable': 'Cable',
            'Port': 'Port',
            'Rack': 'Rack',
            'Switch': 'Switch',
            'fuse': 'Rack'
        }
        
        # Clear previous outputs and prepare fresh output folders
        import shutil
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        
        for folder in set(TARGET_CLASS_MAP.values()):
            os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
        
        # Copy original image to static folder for web access
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_static_path = f"static/segmented_outputs/original_{base_name}.jpg"
        shutil.copy2(image_path, original_static_path)
        
        # Load models
        logging.info("Loading YOLO models...")
        general_model = YOLO(GENERAL_MODEL_PATH)
        port_model = YOLO(PORTS_MODEL_PATH)
        
        # Run general detection
        logging.info("Running general detection...")
        general_results = general_model(img, conf=CONF_THRESHOLD, verbose=False)[0]
        general_class_names = general_model.names
        
        coordinates_data = {}
        object_counter = 0
        region_counter = 0
        segmented_images = []
        
        def crop_and_save(region, path):
            h, w = region.shape[:2]
            if h < MIN_DIM or w < MIN_DIM:
                region = cv2.resize(region, (max(w, MIN_DIM), max(h, MIN_DIM)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, region, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # Process detections
        for box, cls_id, conf in zip(general_results.boxes.xyxy, general_results.boxes.cls, general_results.boxes.conf):
            class_name = general_class_names.get(int(cls_id), f"Unknown_{int(cls_id)}")
            if class_name not in TARGET_CLASS_MAP:
                logging.info(f"Skipping unknown class: {class_name}")
                continue
            
            logging.info(f"[{object_counter}] Found: {class_name} ({float(conf):.2f})")
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)
            x2 = min(img.shape[1], x2 + MARGIN)
            y2 = min(img.shape[0], y2 + MARGIN)
            crop = img[y1:y2, x1:x2]
            
            save_class = TARGET_CLASS_MAP[class_name]
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{class_name}_{object_counter}.jpg"
            out_path = os.path.join(OUTPUT_DIR, save_class, filename)
            crop_and_save(crop, out_path)
            
            rel_path = out_path.replace("\\", "/")
            coordinates_data[rel_path] = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": float(conf), "class_name": class_name
            }
            
            segmented_images.append({
                'path': rel_path,
                'class': save_class,
                'original_class': class_name,
                'confidence': float(conf),
                'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            })
            
            # Detect ports inside Switch or Rack
            if save_class in ['Switch', 'Rack']:
                logging.info(f"Running port detection inside {class_name}")
                port_results = port_model(crop, conf=CONF_THRESHOLD, verbose=False)[0]
                
                for port_idx, (port_box, port_conf) in enumerate(zip(port_results.boxes.xyxy, port_results.boxes.conf)):
                    px1, py1, px2, py2 = map(int, port_box)
                    px1 = max(0, px1 - MARGIN)
                    py1 = max(0, py1 - MARGIN)
                    px2 = min(crop.shape[1], px2 + MARGIN)
                    py2 = min(crop.shape[0], py2 + MARGIN)
                    port_crop = crop[py1:py2, px1:px2]
                    
                    port_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_Port_{region_counter}_{port_idx}.jpg"
                    port_path = os.path.join(OUTPUT_DIR, "Port", port_filename)
                    crop_and_save(port_crop, port_path)
                    
                    rel_port_path = port_path.replace("\\", "/")
                    coordinates_data[rel_port_path] = {
                        "x1": x1 + px1, "y1": y1 + py1,
                        "x2": x1 + px2, "y2": y1 + py2,
                        "width": px2 - px1, "height": py2 - py1,
                        "confidence": float(port_conf),
                        "class_name": "Port",
                        "parent_region": f"{class_name}_{region_counter}"
                    }
                    
                    segmented_images.append({
                        'path': rel_port_path,
                        'class': 'Port',
                        'original_class': 'Port',
                        'confidence': float(port_conf),
                        'coordinates': {'x1': x1 + px1, 'y1': y1 + py1, 'x2': x1 + px2, 'y2': y1 + py2},
                        'parent': f"{class_name}_{region_counter}"
                    })
                
                region_counter += 1
            
            object_counter += 1
        
        # Save coordinates
        os.makedirs(os.path.dirname(COORDINATES_FILE), exist_ok=True)
        with open(COORDINATES_FILE, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        
        # Group images by class for better organization
        grouped_images = {}
        for img_data in segmented_images:
            class_name = img_data['class']
            if class_name not in grouped_images:
                grouped_images[class_name] = []
            grouped_images[class_name].append(img_data)
        
        logging.info(f"Segmentation complete. Found {object_counter} components")
        
        # Generate embeddings and comparisons
        comparison_results = None
        try:
            from utils.embedding_comparison import compare_with_catalog
            
            # Compare with catalog embeddings
            logging.info("Comparing with catalog embeddings...")
            comparison_results = compare_with_catalog(OUTPUT_DIR)
            
        except Exception as e:
            logging.error(f"Error in embeddings/comparison: {str(e)}")
            comparison_results = None
        
        return {
            'success': True,
            'total_components': object_counter,
            'segmented_images': segmented_images,
            'grouped_images': grouped_images,
            'coordinates_file': COORDINATES_FILE,
            'original_image': '/' + original_static_path,
            'comparison_results': comparison_results
        }
        
    except Exception as e:
        logging.error(f"Error in process_image: {str(e)}")
        return {'success': False, 'error': str(e)}
