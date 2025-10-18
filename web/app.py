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
app.config['OUTPUT_DIR'] = Path(__file__).parent.parent / 'data' / 'outputs'
app.config['JSON_SORT_KEYS'] = False

# Import data loader utilities
from utils.data_loader import (
    scan_all_outputs,
    get_model_statistics,
    get_domain_statistics,
    get_inference_details,
    get_comparison_data
)


@app.route('/')
def index():
    """Main dashboard page showing overview of all results."""
    try:
        # Scan all outputs
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Get statistics
        model_stats = get_model_statistics(all_results)
        domain_stats = get_domain_statistics(all_results)
        
        # Calculate overview stats
        total_inferences = len(all_results)
        total_models = len(model_stats)
        total_domains = len(domain_stats)
        success_count = sum(1 for r in all_results if r.get('success', False))
        success_rate = (success_count / total_inferences * 100) if total_inferences > 0 else 0
        
        overview = {
            'total_inferences': total_inferences,
            'total_models': total_models,
            'total_domains': total_domains,
            'success_count': success_count,
            'success_rate': round(success_rate, 1)
        }
        
        return render_template(
            'index.html',
            overview=overview,
            model_stats=model_stats,
            domain_stats=domain_stats,
            recent_results=all_results[:20]  # Show 20 most recent
        )
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/model/<model_name>')
def model_view(model_name):
    """View all results for a specific model."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Filter by model
        model_results = [r for r in all_results if r.get('model') == model_name]
        
        if not model_results:
            return render_template('error.html', error=f'No results found for model: {model_name}'), 404
        
        # Get domain breakdown for this model
        domain_breakdown = {}
        for result in model_results:
            domain = result.get('domain', 'unknown')
            if domain not in domain_breakdown:
                domain_breakdown[domain] = {'total': 0, 'success': 0, 'failed': 0}
            domain_breakdown[domain]['total'] += 1
            if result.get('success', False):
                domain_breakdown[domain]['success'] += 1
            else:
                domain_breakdown[domain]['failed'] += 1
        
        return render_template(
            'model.html',
            model_name=model_name,
            results=model_results,
            domain_breakdown=domain_breakdown,
            total_results=len(model_results)
        )
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/domain/<domain_name>')
def domain_view(domain_name):
    """View all results for a specific domain."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Filter by domain
        domain_results = [r for r in all_results if r.get('domain') == domain_name]
        
        if not domain_results:
            return render_template('error.html', error=f'No results found for domain: {domain_name}'), 404
        
        # Get model breakdown for this domain
        model_breakdown = {}
        for result in domain_results:
            model = result.get('model', 'unknown')
            if model not in model_breakdown:
                model_breakdown[model] = {'total': 0, 'success': 0, 'failed': 0}
            model_breakdown[model]['total'] += 1
            if result.get('success', False):
                model_breakdown[model]['success'] += 1
            else:
                model_breakdown[model]['failed'] += 1
        
        return render_template(
            'domain.html',
            domain_name=domain_name,
            results=domain_results,
            model_breakdown=model_breakdown,
            total_results=len(domain_results)
        )
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/task/<task_id>')
def task_view(task_id):
    """View all results for a specific task across all models."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Filter by task_id
        task_results = [r for r in all_results if r.get('task_id') == task_id]
        
        if not task_results:
            return render_template('error.html', error=f'No results found for task: {task_id}'), 404
        
        # Get the task metadata from the first result
        task_metadata = task_results[0].get('question_metadata', {})
        
        return render_template(
            'task.html',
            task_id=task_id,
            task_metadata=task_metadata,
            results=task_results,
            total_results=len(task_results)
        )
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/inference/<path:inference_id>')
def inference_view(inference_id):
    """View detailed results for a specific inference."""
    try:
        details = get_inference_details(app.config['OUTPUT_DIR'], inference_id)
        
        if not details:
            return render_template('error.html', error=f'Inference not found: {inference_id}'), 404
        
        return render_template(
            'inference.html',
            inference_id=inference_id,
            details=details
        )
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/compare')
def compare_view():
    """Compare results across models and tasks."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        comparison_data = get_comparison_data(all_results)
        
        return render_template(
            'compare.html',
            comparison_data=comparison_data
        )
    except Exception as e:
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


@app.route('/api/statistics')
def api_statistics():
    """API endpoint to get statistics."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        model_stats = get_model_statistics(all_results)
        domain_stats = get_domain_statistics(all_results)
        
        return jsonify({
            'models': model_stats,
            'domains': domain_stats,
            'total_inferences': len(all_results)
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

