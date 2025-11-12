"""
API Service integration example for Simplus EDA Framework.

This example demonstrates how to expose the EDA framework
as a REST API service using Flask.
"""

from flask import Flask, request, jsonify, send_file
import pandas as pd
import io
from pathlib import Path
from datetime import datetime
from simplus_eda import EDAAnalyzer, ReportGenerator
from simplus_eda.core.config import EDAConfig

# Initialize Flask app
app = Flask(__name__)

# Initialize EDA components
analyzer = EDAAnalyzer()
report_generator = ReportGenerator()

# Output directory for reports
OUTPUT_DIR = Path("output/api_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Simplus EDA API',
        'version': '0.1.0'
    })


@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Analyze uploaded data file.

    Expects:
        - File upload with key 'file'
        - Optional JSON config in request form

    Returns:
        JSON with analysis results
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Read file into DataFrame
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(file.read().decode('utf-8')))
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or JSON'}), 400

        # Get optional config
        config = None
        if 'config' in request.form:
            import json
            config = json.loads(request.form['config'])

        # Initialize analyzer with config
        if config:
            analyzer = EDAAnalyzer(config=config)
        else:
            analyzer = EDAAnalyzer()

        # Perform analysis
        results = analyzer.analyze(data)

        return jsonify({
            'status': 'success',
            'results': results,
            'metadata': {
                'rows': data.shape[0],
                'columns': data.shape[1],
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/analyze/report', methods=['POST'])
def analyze_and_generate_report():
    """
    Analyze data and generate downloadable report.

    Expects:
        - File upload with key 'file'
        - Optional 'format' parameter ('html', 'json', 'pdf')

    Returns:
        Downloadable report file
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        report_format = request.form.get('format', 'html')

        # Read file
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(file.read().decode('utf-8')))
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Perform analysis
        results = analyzer.analyze(data)

        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"eda_report_{timestamp}.{report_format}"
        report_path = OUTPUT_DIR / report_filename

        if report_format == 'html':
            report_generator.generate_html(results, str(report_path))
        elif report_format == 'json':
            report_generator.generate_json(results, str(report_path))
        elif report_format == 'pdf':
            report_generator.generate_pdf(results, str(report_path))
        else:
            return jsonify({'error': 'Invalid format. Use html, json, or pdf'}), 400

        # Return file
        return send_file(
            str(report_path),
            as_attachment=True,
            download_name=report_filename
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/analyze/batch', methods=['POST'])
def batch_analyze():
    """
    Batch analyze multiple datasets.

    Expects:
        - Multiple file uploads
        - Optional shared config

    Returns:
        JSON with analysis results for all files
    """
    try:
        files = request.files.getlist('files')

        if not files:
            return jsonify({'error': 'No files provided'}), 400

        results_list = []

        for file in files:
            # Read file
            if file.filename.endswith('.csv'):
                data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            else:
                continue

            # Analyze
            results = analyzer.analyze(data)

            results_list.append({
                'filename': file.filename,
                'results': results,
                'metadata': {
                    'rows': data.shape[0],
                    'columns': data.shape[1]
                }
            })

        return jsonify({
            'status': 'success',
            'count': len(results_list),
            'analyses': results_list,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/config', methods=['GET'])
def get_default_config():
    """Get default configuration options."""
    config = EDAConfig()
    return jsonify({
        'default_config': config.to_dict(),
        'description': 'Default configuration for EDA analysis'
    })


def main():
    """Run the API service."""
    print("=" * 60)
    print("Simplus EDA Framework - API Service")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /analyze             - Analyze uploaded data")
    print("  POST /analyze/report      - Generate downloadable report")
    print("  POST /analyze/batch       - Batch analyze multiple files")
    print("  GET  /config              - Get default configuration")
    print("\n" + "=" * 60)
    print("Starting server on http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
