#!/usr/bin/env python3
"""
VMEvalKit Web Dashboard
A web application to visualize and explore video generation results.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request
from datetime import datetime
from typing import Dict, List, Any, Optional

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vmevalkit-dashboard-secret-key'
app.config['OUTPUT_DIR'] = Path(__file__).parent.parent / 'data' / 'outputs' / 'pilot_experiment'
app.config['JSON_SORT_KEYS'] = False

# Import data loader utilities
from utils.data_loader import (
    scan_all_outputs,
    get_hierarchical_data
)


@app.route('/')
def index():
    """Main dashboard page showing hierarchical view: Models → Domains → Tasks."""
    try:
        # Scan all outputs
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Get hierarchical organization
        hierarchy = get_hierarchical_data(all_results)
        
        # Calculate overview stats
        total_inferences = len(all_results)
        total_models = len(hierarchy)
        all_domains = set()
        for model_domains in hierarchy.values():
            all_domains.update(model_domains.keys())
        total_domains = len(all_domains)
        
        overview = {
            'total_inferences': total_inferences,
            'total_models': total_models,
            'total_domains': total_domains
        }
        
        return render_template(
            'index.html',
            overview=overview,
            hierarchy=hierarchy
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=str(e)), 500


@app.route('/api/results')
def api_results():
    """API endpoint to get all results as JSON."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Filter by query parameters
        model = request.args.get('model')
        domain = request.args.get('domain')
        task_id = request.args.get('task_id')
        
        filtered_results = all_results
        if model:
            filtered_results = [r for r in filtered_results if r.get('model') == model]
        if domain:
            filtered_results = [r for r in filtered_results if r.get('domain') == domain]
        if task_id:
            filtered_results = [r for r in filtered_results if r.get('task_id') == task_id]
        
        return jsonify({
            'total': len(filtered_results),
            'results': filtered_results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/video/<path:video_path>')
def serve_video(video_path):
    """Serve video files from the output directory."""
    try:
        # Construct full path
        full_path = app.config['OUTPUT_DIR'] / video_path
        if not full_path.exists():
            return "Video not found", 404
        
        # Get directory and filename
        directory = full_path.parent
        filename = full_path.name
        
        return send_from_directory(directory, filename, mimetype='video/mp4')
    except Exception as e:
        return str(e), 500


@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve image files from the output directory."""
    try:
        # Construct full path
        full_path = app.config['OUTPUT_DIR'] / image_path
        if not full_path.exists():
            # Try questions directory as fallback
            full_path = Path(__file__).parent.parent / 'data' / 'questions' / image_path
            if not full_path.exists():
                return "Image not found", 404
        
        # Get directory and filename
        directory = full_path.parent
        filename = full_path.name
        
        return send_from_directory(directory, filename)
    except Exception as e:
        return str(e), 500


@app.template_filter('datetime_format')
def datetime_format(value, format='%Y-%m-%d %H:%M:%S'):
    """Format datetime strings."""
    if not value:
        return 'N/A'
    try:
        if isinstance(value, str):
            # Try to parse ISO format
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        else:
            dt = value
        return dt.strftime(format)
    except:
        return value


@app.template_filter('duration_format')
def duration_format(seconds):
    """Format duration in seconds to human-readable format."""
    if not seconds:
        return 'N/A'
    try:
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    except:
        return str(seconds)


@app.template_filter('file_size')
def file_size_format(path):
    """Get and format file size."""
    try:
        if not path or not Path(path).exists():
            return 'N/A'
        size_bytes = Path(path).stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except:
        return 'N/A'


if __name__ == '__main__':
    print("=" * 60)
    print("🎥 VMEvalKit Web Dashboard")
    print("=" * 60)
    print(f"Output directory: {app.config['OUTPUT_DIR']}")
    print(f"Starting server at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

