import os
import json
import logging
from PIL import Image
import shutil
from utils.cable_port_lookup import get_cable_port_connections

try:
    from utils.detailed_switch_analyzer import DetailedSwitchPortAnalyzer
    DETAILED_ANALYSIS_AVAILABLE = True
except ImportError:
    DETAILED_ANALYSIS_AVAILABLE = False
    logging.warning("Detailed switch analyzer not available in mock mode")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Mock segmentation function that creates a working demo
    while we resolve the YOLO model dependencies
    """
    try:
        # Check if image exists and is valid
        if not os.path.exists(image_path):
            return {'success': False, 'error': f'Image not found: {image_path}'}
        
        # Try to open the image to validate it
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            return {'success': False, 'error': f'Invalid image file: {str(e)}'}
        
        OUTPUT_DIR = "static/segmented_outputs"
        COORDINATES_FILE = "static/segmented_outputs/coordinates.json"
        
        # Create output directories
        categories = ['Rack', 'Switch', 'Port', 'Cable']
        for category in categories:
            os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)
        
        # Copy original image to static folder for web access
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_static_path = f"static/segmented_outputs/original_{base_name}.jpg"
        shutil.copy2(image_path, original_static_path)
        
        # Create demo segmented images by copying the original
        segmented_images = []
        coordinates_data = {}
        
        # Create mock detections for demo
        mock_detections = [
            {'class': 'Rack', 'confidence': 0.85, 'coords': [50, 50, 200, 300]},
            {'class': 'Switch', 'confidence': 0.92, 'coords': [220, 100, 400, 180]},
            {'class': 'Cable', 'confidence': 0.78, 'coords': [100, 320, 350, 380]},
            {'class': 'Port', 'confidence': 0.89, 'coords': [250, 120, 280, 140]}
        ]
        
        for i, detection in enumerate(mock_detections):
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['coords']
            
            # Create output filename
            filename = f"{base_name}_{class_name}_{i}.jpg"
            out_path = os.path.join(OUTPUT_DIR, class_name, filename)
            
            # Copy original image as placeholder (in real implementation, this would be the cropped segment)
            shutil.copy2(image_path, out_path)
            
            rel_path = out_path.replace("\\", "/")
            coordinates_data[rel_path] = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": confidence,
                "class_name": class_name
            }
            
            segmented_images.append({
                'path': rel_path,
                'class': class_name,
                'original_class': class_name,
                'confidence': confidence,
                'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            })
        
        # Save coordinates
        os.makedirs(os.path.dirname(COORDINATES_FILE), exist_ok=True)
        with open(COORDINATES_FILE, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        
        # Group images by class
        grouped_images = {}
        for img_data in segmented_images:
            class_name = img_data['class']
            if class_name not in grouped_images:
                grouped_images[class_name] = []
            grouped_images[class_name].append(img_data)
        
        logging.info(f"Demo segmentation complete. Created {len(segmented_images)} mock detections")
        
        # Create mock comparison results for demo
        comparison_results = []
        for img_data in segmented_images:
            result = {
                'cropped_image': img_data['path'],
                'category': img_data['class'],
                'name': f"Sample {img_data['class']} Component",
                'description': f"This is a demo {img_data['class'].lower()} component detected in your network infrastructure image.",
                'similarity_score': 0.85 + (len(comparison_results) * 0.02),  # Vary scores slightly
                'matched_image': None,  # No catalog images in demo mode
                'coordinates': img_data['coordinates']
            }
            
            # Add cable port information for cable components
            if img_data['class'].lower() == 'cable':
                cable_name = result['name']
                port_connections = get_cable_port_connections(cable_name)
                if port_connections:
                    result['cable_port_info'] = port_connections
            
            # Add detailed switch analysis for switch components
            elif img_data['class'].lower() == 'switch':
                result['switch_analysis'] = _create_mock_switch_analysis()
            
            comparison_results.append(result)

        return {
            'success': True,
            'total_components': len(segmented_images),
            'segmented_images': segmented_images,
            'grouped_images': grouped_images,
            'coordinates_file': COORDINATES_FILE,
            'original_image': '/' + original_static_path,
            'comparison_results': comparison_results,
            'demo_mode': True
        }
        
    except Exception as e:
        logging.error(f"Error in mock process_image: {str(e)}")
        return {'success': False, 'error': str(e)}

def _create_mock_switch_analysis():
    """Create realistic mock switch analysis data"""
    import random
    
    # Generate realistic port data
    total_ports = random.choice([24, 48])
    connected_ports = random.randint(int(total_ports * 0.4), int(total_ports * 0.8))
    empty_ports = total_ports - connected_ports
    utilization_rate = round((connected_ports / total_ports) * 100, 1)
    
    # Generate cable color distribution
    cable_colors = ['blue', 'yellow', 'green', 'red', 'orange', 'purple', 'teal', 'black']
    cable_distribution = {}
    remaining_connected = connected_ports
    
    for i, color in enumerate(cable_colors):
        if remaining_connected <= 0:
            break
        if i == len(cable_colors) - 1:  # Last color gets remaining
            count = remaining_connected
        else:
            max_count = min(remaining_connected, random.randint(1, 8))
            count = random.randint(0, max_count)
        
        if count > 0:
            cable_distribution[color] = count
            remaining_connected -= count
    
    # Generate LED status distribution
    led_distribution = {
        'active': random.randint(int(connected_ports * 0.6), connected_ports),
        'link': random.randint(0, int(connected_ports * 0.2)),
        'inactive': empty_ports
    }
    
    # Add some activity/error LEDs
    if random.random() > 0.7:
        led_distribution['activity'] = random.randint(1, 3)
    if random.random() > 0.9:
        led_distribution['error'] = random.randint(1, 2)
    
    # Generate individual port data
    ports = []
    port_num = 1
    
    # Connected ports with cables
    for color, count in cable_distribution.items():
        for _ in range(count):
            ports.append({
                'port_number': port_num,
                'bbox': [10 + (port_num % 12) * 20, 10 + (port_num // 12) * 15, 18, 12],
                'has_cable': True,
                'cable_color': color,
                'cable_confidence': round(random.uniform(0.7, 0.95), 3),
                'led_status': random.choice(['active', 'link', 'activity']),
                'led_confidence': round(random.uniform(0.6, 0.9), 3)
            })
            port_num += 1
    
    # Empty ports
    while port_num <= total_ports:
        ports.append({
            'port_number': port_num,
            'bbox': [10 + (port_num % 12) * 20, 10 + (port_num // 12) * 15, 18, 12],
            'has_cable': False,
            'cable_color': None,
            'cable_confidence': 0.0,
            'led_status': 'inactive',
            'led_confidence': 0.8
        })
        port_num += 1
    
    return {
        'switch_info': {
            'name': 'Demo Switch',
            'analysis_method': 'mock_detailed_analysis'
        },
        'summary': {
            'total_ports': total_ports,
            'connected_ports': connected_ports,
            'empty_ports': empty_ports,
            'utilization_rate': utilization_rate,
            'cable_distribution': cable_distribution,
            'led_distribution': led_distribution
        },
        'ports': ports
    }