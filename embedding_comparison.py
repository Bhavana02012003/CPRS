import numpy as np
import pandas as pd
import faiss
import pickle
import os
import json
import logging
from cropped_embeddings import generate_category_embeddings
from utils.switch_analyzer import analyze_switch_image, format_cable_distribution, format_led_status
from utils.cable_port_lookup import get_cable_port_connections

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def get_coordinates_for_image(cropped_img_path, coordinates_data):
    coords = {}
    # Try exact path first
    if cropped_img_path in coordinates_data:
        coords = coordinates_data[cropped_img_path]
    else:
        # Try without leading slash
        path_without_slash = cropped_img_path.lstrip('/')
        if path_without_slash in coordinates_data:
            coords = coordinates_data[path_without_slash]
        else:
            # Try with leading slash
            path_with_slash = '/' + cropped_img_path.lstrip('/')
            if path_with_slash in coordinates_data:
                coords = coordinates_data[path_with_slash]
            else:
                # Try filename matching as last resort
                filename = os.path.basename(cropped_img_path)
                for key in coordinates_data.keys():
                    if os.path.basename(key) == filename:
                        coords = coordinates_data[key]
                        break
    return coords

def clean_cropped_path(cropped_img_path):
    if 'segmented_outputs' in cropped_img_path:
        parts = cropped_img_path.split('segmented_outputs')
        if len(parts) > 1:
            cropped_img_path = '/static/segmented_outputs' + parts[1].replace('\\', '/')
        else:
            cropped_img_path = cropped_img_path.replace('\\', '/')
            if ':' in cropped_img_path:
                cropped_img_path = '/' + '/'.join(cropped_img_path.split('/')[-3:])
    return cropped_img_path

def compare_with_catalog(segmented_outputs_dir):
    """
    Compare segmented images with catalog embeddings and return results for card display
    """
    try:
        catalog_path = "all_categories_data.pkl"
        metadata_path = "metadata.pkl"
        
        # Check if catalog files exist
        if not os.path.exists(catalog_path):
            logging.warning(f"Catalog embeddings file not found: {catalog_path}")
            return None
        
        if not os.path.exists(metadata_path):
            logging.warning(f"Metadata file not found: {metadata_path}")
            return None
        
        # Generate embeddings for cropped images
        logging.info("Generating embeddings for segmented images...")
        cropped_data = generate_category_embeddings(segmented_outputs_dir)
        
        if not cropped_data:
            logging.warning("No cropped embeddings generated")
            return None
        
        # Load catalog data
        logging.info("Loading catalog embeddings and metadata...")
        with open(catalog_path, "rb") as f:
            catalog_data = pickle.load(f)
        
        metadata = pd.read_pickle(metadata_path)
        
        # Load coordinates data
        coordinates_data = {}
        coordinates_file = "static/segmented_outputs/coordinates.json"
        if os.path.exists(coordinates_file):
            with open(coordinates_file, 'r') as f:
                coordinates_data = json.load(f)
        
        results = []
        
        # Collect all cropped images
        all_cropped_images = []
        all_categories = set(cropped_data.keys())
        
        for category in sorted(all_categories):
            crop_paths = cropped_data[category]["image_paths"]
            for path in crop_paths:
                cleaned_path = clean_cropped_path(path)
                all_cropped_images.append({
                    'path': cleaned_path,
                    'category': category,
                    'original_path': path
                })
        
        # Process each cropped image
        for img_info in all_cropped_images:
            cropped_img_path = img_info['path']
            category = img_info['category']
            
            # Get coordinates for this image
            coords = get_coordinates_for_image(cropped_img_path, coordinates_data)
            
            result_data = {
                'cropped_image': cropped_img_path,
                'category': category,
                'name': 'No match found',
                'description': 'Component detected but no catalog match found',
                'matched_image': '',
                'similarity_score': 0.0,
                'coordinates': coords
            }
            
            # Add dynamic switch analysis for switch components
            if category.lower() == 'switch':
                # Get the actual image path for analysis
                actual_image_path = img_info['original_path']
                if actual_image_path.startswith('static/'):
                    actual_image_path = actual_image_path
                elif not actual_image_path.startswith('/'):
                    actual_image_path = f"static/{actual_image_path}"
                
                switch_analysis = analyze_switch_image(actual_image_path)
                if switch_analysis:
                    result_data['switch_analysis'] = switch_analysis
            
            # Check if we have catalog data for this category
            if category in catalog_data and category in metadata:
                crop_embeds = cropped_data[category]["image_embeddings"]
                crop_paths = cropped_data[category]["image_paths"]
                cat_embeds = catalog_data[category]["image_embeddings"]
                
                # Find the index of this specific image
                img_index = None
                for i, path in enumerate(crop_paths):
                    if clean_cropped_path(path) == cropped_img_path:
                        img_index = i
                        break
                
                if img_index is not None and len(cat_embeds) > 0:
                    # Get embedding for this specific image
                    crop_embed = l2_normalize(np.array([crop_embeds[img_index]]))
                    cat_embeds_norm = l2_normalize(np.array(cat_embeds))
                    
                    # Find best match using FAISS
                    dim = cat_embeds_norm.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    index.add(cat_embeds_norm)
                    distances, indices = index.search(crop_embed, 1)
                    
                    if len(indices[0]) > 0:
                        best_idx = indices[0][0]
                        best_score = float(distances[0][0])
                        meta_row = metadata[category].iloc[best_idx]
                        
                        # Clean catalog image path
                        matched_img = meta_row['Image']
                        if isinstance(matched_img, str) and 'static' in matched_img:
                            matched_img = '/' + matched_img.split('static', 1)[1].replace('\\', '/')
                        
                        result_data.update({
                            'name': meta_row['Name'],
                            'description': meta_row['Description'],
                            'matched_image': matched_img,
                            'similarity_score': best_score
                        })
                        
                        # Add cable port information for cable components
                        if category.lower() == 'cable':
                            cable_name = meta_row['Name']
                            port_connections = get_cable_port_connections(cable_name)
                            if port_connections:
                                result_data['cable_port_info'] = port_connections
            
            # If no catalog match found but it's a cable, still try to get port info
            if category.lower() == 'cable' and 'cable_port_info' not in result_data:
                # Try with the detected component name if available
                if result_data.get('name') and result_data['name'] != 'No match found':
                    port_connections = get_cable_port_connections(result_data['name'])
                    if port_connections:
                        result_data['cable_port_info'] = port_connections
            
            results.append(result_data)
        
        logging.info(f"Comparison complete. Processed {len(results)} components")
        return results
        
    except Exception as e:
        logging.error(f"Error in compare_with_catalog: {str(e)}")
        return None