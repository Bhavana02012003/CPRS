import cv2
import numpy as np
import json
import os
from datetime import datetime
import logging

class DetailedSwitchPortAnalyzer:
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        
        self.cable_colors = {
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            'blue': {'lower': np.array([100, 80, 80]), 'upper': np.array([130, 255, 255])},
            'teal': {'lower': np.array([80, 80, 80]), 'upper': np.array([95, 255, 255])},
            'cyan': {'lower': np.array([85, 80, 80]), 'upper': np.array([105, 255, 255])},
            'purple': {'lower': np.array([130, 80, 80]), 'upper': np.array([160, 255, 255])},
            'green': {'lower': np.array([40, 80, 80]), 'upper': np.array([75, 255, 255])},
            'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([20, 255, 255])},
            'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])},
            'white': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            'gray': {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 30, 150])},
        }
        
        self.led_colors = {
            'green': {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])},
            'orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([25, 255, 255])},
            'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
        }
        
        self.min_color_coverage = 0.15
        self.min_combined_score = 0.6
    
    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def analyze_switch_image(self, image_path, switch_name="Unknown"):
        logging.info(f"Analyzing detailed switch ports: {switch_name}")
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not load image {image_path}")
                return None
            
            h, w = image.shape[:2]
            logging.info(f"Image dimensions: {w}x{h}")
            
            port_regions = self._comprehensive_port_detection(image)
            logging.info(f"Detected {len(port_regions)} port regions")
            
            port_analyses = []
            for i, region in enumerate(port_regions):
                port_analysis = self._thorough_port_analysis(image, region, i + 1)
                port_analyses.append(port_analysis)
            
            results = self._generate_comprehensive_results(image, port_analyses, switch_name)
            results = self._convert_to_json_serializable(results)
            
            connected = results['summary']['connected_ports']
            empty = results['summary']['empty_ports']
            total = results['summary']['total_ports']
            utilization = results['summary']['utilization_rate']
            
            logging.info(f"Analysis complete: {total} ports ({connected} connected, {empty} empty, {utilization:.1f}%)")
            
            return results
            
        except Exception as e:
            logging.error(f"Error analyzing switch image: {str(e)}")
            return None
    
    def _comprehensive_port_detection(self, image):
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        port_regions = self._edge_based_port_detection(image)
        
        if len(port_regions) < 12:
            logging.info("Using grid detection fallback")
            port_regions = self._comprehensive_grid_detection(image)
        
        return port_regions
    
    def _edge_based_port_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        
        edges = cv2.bitwise_or(edges1, edges2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        port_candidates = []
        min_area = 200
        max_area = (w * h) // 20
        
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            area = w_rect * h_rect
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            
            if (min_area < area < max_area and
                0.3 < aspect_ratio < 4.0 and
                x > 2 and y > 2 and 
                x + w_rect < w - 2 and y + h_rect < h - 2):
                
                port_candidates.append({
                    'bbox': (x, y, w_rect, h_rect),
                    'area': area,
                    'score': area / 1000.0
                })
        
        filtered_ports = self._remove_overlapping_regions(port_candidates)
        
        return filtered_ports[:50]
    
    def _comprehensive_grid_detection(self, image):
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 7:
            rows, cols = 2, 24
        elif aspect_ratio > 5:
            if h > 80:
                rows, cols = 2, 24
            else:
                rows, cols = 1, 24
        elif aspect_ratio > 3:
            rows, cols = 1, 24
        else:
            rows, cols = 1, 12
        
        logging.info(f"Grid layout: {rows}x{cols} = {rows*cols} ports")
        
        margin_x = max(5, w // 50)
        margin_y = max(5, h // 30)
        
        usable_w = w - 2 * margin_x
        usable_h = h - 2 * margin_y
        
        port_w = usable_w / cols
        port_h = usable_h / rows
        
        port_regions = []
        
        for row in range(rows):
            for col in range(cols):
                x = int(margin_x + col * port_w)
                y = int(margin_y + row * port_h)
                
                w_port = int(port_w * 0.9)
                h_port = int(port_h * 0.9)
                
                if w_port > 8 and h_port > 8:
                    port_regions.append({
                        'bbox': (x, y, w_port, h_port),
                        'area': w_port * h_port,
                        'score': 1.0,
                        'port_number': row * cols + col + 1
                    })
        
        return port_regions
    
    def _remove_overlapping_regions(self, candidates):
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        filtered = []
        for candidate in candidates:
            x, y, w, h = candidate['bbox']
            
            overlap_found = False
            for existing in filtered:
                ex, ey, ew, eh = existing['bbox']
                
                ix1, iy1 = max(x, ex), max(y, ey)
                ix2, iy2 = min(x + w, ex + ew), min(y + h, ey + eh)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1, area2 = w * h, ew * eh
                    
                    if intersection > 0.3 * min(area1, area2):
                        overlap_found = True
                        break
            
            if not overlap_found:
                filtered.append(candidate)
        
        return filtered
    
    def _thorough_port_analysis(self, image, port_region, port_number):
        x, y, w, h = port_region['bbox']
        port_roi = image[y:y+h, x:x+w]
        
        if port_roi.size == 0:
            return self._create_empty_port_result(port_region, port_number)
        
        cable_result = self._conservative_cable_detection(port_roi)
        led_result = self._led_status_detection(port_roi)
        
        return {
            'port_number': int(port_number),
            'bbox': [int(coord) for coord in port_region['bbox']],
            'has_cable': bool(cable_result['has_cable']),
            'cable_color': cable_result['color'],
            'cable_confidence': float(cable_result['confidence']),
            'led_status': led_result['status'],
            'led_confidence': float(led_result['confidence']),
            'detection_details': {
                'cable_analysis': cable_result['details'],
                'led_analysis': led_result['details']
            }
        }
    
    def _conservative_cable_detection(self, port_roi):
        if port_roi.size == 0:
            return {'has_cable': False, 'color': None, 'confidence': 0.0, 'details': {}}
        
        try:
            hsv_roi = cv2.cvtColor(port_roi, cv2.COLOR_BGR2HSV)
            gray_roi = cv2.cvtColor(port_roi, cv2.COLOR_BGR2GRAY)
            total_pixels = port_roi.shape[0] * port_roi.shape[1]
            
            color_scores = {}
            best_color = None
            max_color_score = 0.0
            
            for color_name, color_range in self.cable_colors.items():
                mask = cv2.inRange(hsv_roi, color_range['lower'], color_range['upper'])
                color_pixels = cv2.countNonZero(mask)
                score = color_pixels / total_pixels if total_pixels > 0 else 0
                
                color_scores[color_name] = round(float(score), 4)
                
                if score > max_color_score:
                    max_color_score = score
                    best_color = color_name
            
            features = self._detailed_feature_analysis(port_roi, gray_roi, hsv_roi)
            
            has_cable = False
            confidence = 0.0
            
            if max_color_score > self.min_color_coverage:
                supporting_evidence = 0
                
                if features['edge_density'] > 0.03:
                    supporting_evidence += 1
                if features['saturation_mean'] > 50:
                    supporting_evidence += 1
                if features['brightness_std'] > 20:
                    supporting_evidence += 1
                if 40 < features['brightness_mean'] < 180:
                    supporting_evidence += 1
                
                if supporting_evidence >= 2:
                    has_cable = True
                    confidence = max_color_score * 0.7 + (supporting_evidence / 4) * 0.3
            
            elif (features['edge_density'] > 0.08 and 
                  features['saturation_mean'] > 80 and
                  features['brightness_std'] > 40):
                has_cable = True
                confidence = 0.4
                best_color = self._find_secondary_color(color_scores)
            
            return {
                'has_cable': has_cable,
                'color': best_color if has_cable and max_color_score > 0.05 else None,
                'confidence': round(confidence, 3),
                'details': {
                    'color_scores': color_scores,
                    'max_color_score': round(max_color_score, 4),
                    'features': features
                }
            }
        except Exception as e:
            logging.error(f"Error in cable detection: {str(e)}")
            return {'has_cable': False, 'color': None, 'confidence': 0.0, 'details': {'error': str(e)}}
    
    def _find_secondary_color(self, color_scores):
        cable_colors = {k: v for k, v in color_scores.items() 
                       if k not in ['black', 'gray', 'white'] and v > 0.02}
        
        if cable_colors:
            return max(cable_colors.items(), key=lambda x: x[1])[0]
        return None
    
    def _led_status_detection(self, port_roi):
        if port_roi.size == 0:
            return {'status': 'inactive', 'confidence': 0.0, 'details': {}}
        
        try:
            hsv_roi = cv2.cvtColor(port_roi, cv2.COLOR_BGR2HSV)
            total_pixels = port_roi.shape[0] * port_roi.shape[1]
            
            led_detections = {}
            max_led_coverage = 0.0
            dominant_led = None
            
            for led_color, color_range in self.led_colors.items():
                mask = cv2.inRange(hsv_roi, color_range['lower'], color_range['upper'])
                led_pixels = cv2.countNonZero(mask)
                coverage = led_pixels / total_pixels if total_pixels > 0 else 0
                
                led_detections[led_color] = round(float(coverage), 4)
                
                if coverage > max_led_coverage:
                    max_led_coverage = coverage
                    dominant_led = led_color
            
            status = 'inactive'
            confidence = 0.0
            
            if max_led_coverage > 0.02:
                if dominant_led == 'green':
                    status = 'active'
                elif dominant_led == 'orange':
                    status = 'link'
                elif dominant_led == 'red':
                    status = 'error'
                elif dominant_led == 'blue':
                    status = 'activity'
                else:
                    status = 'unknown'
                
                confidence = min(max_led_coverage * 20, 1.0)
            else:
                status = 'inactive'
                confidence = 0.8
            
            return {
                'status': status,
                'confidence': round(confidence, 3),
                'details': {
                    'led_detections': led_detections,
                    'max_coverage': round(max_led_coverage, 4),
                    'dominant_led': dominant_led
                }
            }
        except Exception as e:
            logging.error(f"Error in LED detection: {str(e)}")
            return {'status': 'inactive', 'confidence': 0.0, 'details': {'error': str(e)}}
    
    def _detailed_feature_analysis(self, port_roi, gray_roi, hsv_roi):
        try:
            brightness_mean = float(np.mean(gray_roi))
            brightness_std = float(np.std(gray_roi))
            
            saturation = hsv_roi[:, :, 1]
            saturation_mean = float(np.mean(saturation))
            
            edges = cv2.Canny(gray_roi, 30, 100)
            edge_density = float(np.sum(edges > 0)) / edges.size if edges.size > 0 else 0
            
            return {
                'brightness_mean': round(brightness_mean, 1),
                'brightness_std': round(brightness_std, 1),
                'saturation_mean': round(saturation_mean, 1),
                'edge_density': round(edge_density, 4)
            }
        except Exception as e:
            logging.error(f"Error in feature analysis: {str(e)}")
            return {
                'brightness_mean': 0.0,
                'brightness_std': 0.0,
                'saturation_mean': 0.0,
                'edge_density': 0.0
            }
    
    def _create_empty_port_result(self, port_region, port_number):
        return {
            'port_number': int(port_number),
            'bbox': [int(x) for x in port_region['bbox']],
            'has_cable': False,
            'cable_color': None,
            'cable_confidence': 0.0,
            'led_status': 'inactive',
            'led_confidence': 0.0,
            'detection_details': {'error': 'Empty or invalid port region'}
        }
    
    def _generate_comprehensive_results(self, image, port_analyses, switch_name):
        total_ports = len(port_analyses)
        connected_ports = sum(1 for p in port_analyses if p['has_cable'])
        empty_ports = total_ports - connected_ports
        
        cable_distribution = {}
        for port in port_analyses:
            if port['has_cable'] and port['cable_color']:
                color = port['cable_color']
                cable_distribution[color] = cable_distribution.get(color, 0) + 1
        
        led_distribution = {}
        for port in port_analyses:
            status = port.get('led_status', 'inactive')
            led_distribution[status] = led_distribution.get(status, 0) + 1
        
        return {
            'switch_info': {
                'name': switch_name,
                'image_dimensions': [int(x) for x in image.shape],
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_method': 'detailed_computer_vision'
            },
            'summary': {
                'total_ports': int(total_ports),
                'connected_ports': int(connected_ports),
                'empty_ports': int(empty_ports),
                'utilization_rate': round(float(connected_ports / total_ports * 100) if total_ports > 0 else 0, 1),
                'cable_distribution': cable_distribution,
                'led_distribution': led_distribution
            },
            'ports': port_analyses
        }