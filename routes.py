import os
import json
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from app import app
try:
    from utils.segmentation import process_image, allowed_file
except ImportError:
    # Use mock segmentation while YOLO dependencies are unavailable
    from utils.mock_segmentation import process_image, allowed_file
import zipfile
import io

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            import time
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Process the image
            logging.info(f"Processing image: {filepath}")
            results = process_image(filepath)
            
            if results['success']:
                return render_template('results.html', 
                                     results=results, 
                                     original_filename=filename)
            else:
                flash(f'Error processing image: {results["error"]}', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            logging.error(f"Error processing upload: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload JPG, JPEG, PNG, or BMP images.', 'error')
        return redirect(url_for('index'))

@app.route('/download_results/<path:filename>')
def download_results(filename):
    """Download individual segmented images"""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading file {filename}: {str(e)}")
        flash('File not found', 'error')
        return redirect(url_for('index'))

@app.route('/download_all_results')
def download_all_results():
    """Download all segmented results as a ZIP file"""
    try:
        # Get the latest coordinates file
        coordinates_file = os.path.join(app.config['SEGMENTED_OUTPUTS'], 'coordinates.json')
        
        if not os.path.exists(coordinates_file):
            flash('No results to download', 'error')
            return redirect(url_for('index'))
        
        # Create a ZIP file in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add coordinates file
            zf.write(coordinates_file, 'coordinates.json')
            
            # Add all segmented images
            with open(coordinates_file, 'r') as f:
                coordinates_data = json.load(f)
            
            for image_path in coordinates_data.keys():
                if os.path.exists(image_path):
                    # Use relative path in ZIP
                    arc_name = os.path.relpath(image_path, app.config['SEGMENTED_OUTPUTS'])
                    zf.write(image_path, f"segmented_images/{arc_name}")
        
        memory_file.seek(0)
        
        return send_file(
            io.BytesIO(memory_file.read()),
            mimetype='application/zip',
            as_attachment=True,
            download_name='segmentation_results.zip'
        )
        
    except Exception as e:
        logging.error(f"Error creating ZIP file: {str(e)}")
        flash('Error creating download file', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))
