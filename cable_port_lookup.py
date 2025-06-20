import csv
import os
import logging
import re
from difflib import SequenceMatcher

def normalize_cable_name(name):
    """
    Normalize cable name by removing extra spaces, special characters, and converting to lowercase
    """
    if not name:
        return ""
    # Remove extra spaces, convert to lowercase, remove special characters except alphanumeric and basic punctuation
    normalized = re.sub(r'[^\w\s\-\.]', '', str(name).strip().lower())
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def calculate_similarity(str1, str2):
    """
    Calculate similarity between two strings using SequenceMatcher
    """
    return SequenceMatcher(None, str1, str2).ratio()

def find_best_cable_match(target_name, available_names, threshold=0.6):
    """
    Find the best matching cable name from available names using fuzzy matching
    """
    target_normalized = normalize_cable_name(target_name)
    best_match = None
    best_score = 0
    
    for name in available_names:
        name_normalized = normalize_cable_name(name)
        score = calculate_similarity(target_normalized, name_normalized)
        
        # Also check if one name contains the other (for partial matches)
        if target_normalized in name_normalized or name_normalized in target_normalized:
            score = max(score, 0.8)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = name
    
    logging.info(f"Best match for '{target_name}': '{best_match}' (score: {best_score:.2f})")
    return best_match, best_score

def get_cable_port_connections(cable_name):
    """
    Look up port connections for a given cable name
    
    Args:
        cable_name (str): Name of the cable to look up
        
    Returns:
        dict: Dictionary with port1 and port2 connections, or None if not found
    """
    try:
        # Check for available data files (prefer CSV for reliability, fallback to Excel)
        csv_file = "cables-ports.csv"
        excel_files = ["Cables-ports 1.xlsx", "Cables-ports.xlsx"]
        
        data_file = None
        file_type = None
        
        # Check for CSV file first (more reliable)
        if os.path.exists(csv_file):
            data_file = csv_file
            file_type = 'csv'
        else:
            # Fallback to Excel files
            for excel_file in excel_files:
                if os.path.exists(excel_file):
                    data_file = excel_file
                    file_type = 'excel'
                    break
        
        if not data_file:
            logging.warning(f"Cable-port mapping file not found")
            return None
        
        # Read the data file and collect all cable names for fuzzy matching
        all_cable_data = []
        
        if file_type == 'excel':
            try:
                # Try to import pandas dynamically
                import pandas as pd
                df = pd.read_excel(data_file)
                
                for _, row in df.iterrows():
                    all_cable_data.append({
                        'name': str(row['Name']).strip(),
                        'port1': str(row['Port 1']).strip(),
                        'port2': str(row['Port 2']).strip()
                    })
                    
            except ImportError:
                logging.warning(f"pandas not available, cannot read Excel file {data_file}")
                return None
            except Exception as e:
                logging.error(f"Error reading Excel file {data_file}: {str(e)}")
                return None
                
        elif file_type == 'csv':
            try:
                with open(data_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_cable_data.append({
                            'name': row['Name'].strip(),
                            'port1': row['Port 1'].strip(),
                            'port2': row['Port 2'].strip()
                        })
            except Exception as e:
                logging.error(f"Error reading CSV file {data_file}: {str(e)}")
                return None
        
        # Use fuzzy matching to find the best cable match
        if all_cable_data:
            available_names = [item['name'] for item in all_cable_data]
            best_match, score = find_best_cable_match(cable_name, available_names)
            
            if best_match:
                # Find the data for the best match
                for item in all_cable_data:
                    if item['name'] == best_match:
                        return {
                            'port1': item['port1'],
                            'port2': item['port2'],
                            'cable_name': item['name'],
                            'match_score': score
                        }
            
        logging.info(f"No port connections found for cable: {cable_name}")
        return None
            
    except Exception as e:
        logging.error(f"Error looking up cable port connections: {str(e)}")
        return None

def format_cable_port_info(port_info):
    """
    Format cable port information for display
    
    Args:
        port_info (dict): Port connection information
        
    Returns:
        str: Formatted HTML string for display
    """
    if not port_info:
        return ""
    
    return f"""
    <div class="mt-3 p-3 bg-light rounded">
        <h6 class="text-info mb-2">
            <i class="fas fa-plug me-1"></i>Port Connections
        </h6>
        <div class="row text-center">
            <div class="col-6">
                <div class="h6 mb-0 text-primary">{port_info['port1']}</div>
                <small class="text-muted">Port 1</small>
            </div>
            <div class="col-6">
                <div class="h6 mb-0 text-primary">{port_info['port2']}</div>
                <small class="text-muted">Port 2</small>
            </div>
        </div>
        <div class="text-center mt-2">
            <small class="text-muted">
                <i class="fas fa-arrows-alt-h me-1"></i>
                {port_info['port1']} ↔ {port_info['port2']}
            </small>
        </div>
    </div>
    """