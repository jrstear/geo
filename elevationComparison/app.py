from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from tin_processor import parse_landxml_tin, parse_gcp_csv, compare_elevations, calculate_statistics

# Try to import reportlab for PDF export
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

app = Flask(__name__)
# Increased to 500MB to handle large TIN files (200MB+)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration - supports both local and Cloud Run
# Cloud Run sets PORT env var, local development uses default 5000
PORT = int(os.environ.get('PORT', 5000))
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'

# Upload folder - use /tmp on Cloud Run (ephemeral storage), 'uploads' locally
if os.environ.get('GAE_ENV') or os.environ.get('K_SERVICE'):
    # Running on Cloud Run or App Engine
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
else:
    # Running locally
    app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xml', 'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Debug: Log what files were received
        app.logger.info(f"Files in request: {list(request.files.keys())}")
        
        if 'tin_file' not in request.files or 'gcp_file' not in request.files:
            app.logger.warning("Missing files in request")
            return jsonify({'error': 'Both TIN and GCP files are required'}), 400
        
        tin_file = request.files['tin_file']
        gcp_file = request.files['gcp_file']
        
        app.logger.info(f"TIN filename: {tin_file.filename}, GCP filename: {gcp_file.filename}")
        
        if tin_file.filename == '' or gcp_file.filename == '':
            app.logger.warning("Empty filenames")
            return jsonify({'error': 'Both files must be selected'}), 400
        
        if not allowed_file(tin_file.filename) or not allowed_file(gcp_file.filename):
            app.logger.warning(f"Invalid file types - TIN: {tin_file.filename}, GCP: {gcp_file.filename}")
            return jsonify({
                'error': f'Invalid file type. TIN must be XML and GCP must be CSV. Received: TIN={tin_file.filename}, GCP={gcp_file.filename}'
            }), 400
    except Exception as e:
        app.logger.error(f"Error in upload validation: {str(e)}")
        return jsonify({'error': f'Upload validation error: {str(e)}'}), 400
    
    # Save files temporarily
    tin_path = None
    gcp_path = None
    
    try:
        # Save TIN file
        tin_filename = secure_filename(tin_file.filename)
        tin_path = os.path.join(app.config['UPLOAD_FOLDER'], tin_filename)
        tin_file.save(tin_path)
        
        # Save GCP file
        gcp_filename = secure_filename(gcp_file.filename)
        gcp_path = os.path.join(app.config['UPLOAD_FOLDER'], gcp_filename)
        gcp_file.save(gcp_path)
        
        # Parse files
        try:
            tin_triangles, unit = parse_landxml_tin(tin_path)
        except Exception as e:
            return jsonify({'error': f'Error parsing TIN file: {str(e)}'}), 400
        
        try:
            gcp_data = parse_gcp_csv(gcp_path)
        except Exception as e:
            return jsonify({'error': f'Error parsing GCP file: {str(e)}'}), 400
        
        # Compare elevations
        try:
            # Use debug mode if enabled via environment variable or if running locally in debug
            use_debug = DEBUG_MODE or (not os.environ.get('GAE_ENV') and not os.environ.get('K_SERVICE'))
            app.logger.info(f"Detected unit from TIN file: '{unit}'")
            results = compare_elevations(tin_triangles, gcp_data, debug=use_debug)
            statistics = calculate_statistics(results)
            statistics['unit'] = unit  # Add unit to statistics
            if use_debug:
                app.logger.info(f"Unit set in statistics: '{statistics.get('unit', 'NOT SET')}'")
        except Exception as e:
            import traceback
            app.logger.error(f"Error comparing elevations: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f'Error comparing elevations: {str(e)}'}), 500
        
        # Clean up uploaded files
        if tin_path and os.path.exists(tin_path):
            os.remove(tin_path)
        if gcp_path and os.path.exists(gcp_path):
            os.remove(gcp_path)
        
        # Render results page
        return render_template('results.html', results=results, statistics=statistics)
        
    except Exception as e:
        # Clean up on error
        if tin_path and os.path.exists(tin_path):
            os.remove(tin_path)
        if gcp_path and os.path.exists(gcp_path):
            os.remove(gcp_path)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/export', methods=['POST'])
def export_results():
    """Export results as CSV"""
    import csv
    import io
    from flask import Response
    
    data = request.json
    if not data or 'results' not in data:
        return jsonify({'error': 'No data to export'}), 400
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['GCP ID', 'Easting', 'Northing', 'GCP Elevation', 'TIN Elevation', 
                     'Discrepancy', 'Absolute Error', 'GCP Type'])
    
    # Write data
    for row in data['results']:
        writer.writerow([
            row['gcp_id'],
            row['easting'],
            row['northing'],
            row['gcp_elevation'],
            row['tin_elevation'],
            row['discrepancy'],
            row['absolute_error'],
            row.get('gcp_type', '')
        ])
    
    output.seek(0)
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=elevation_comparison_results.csv'}
    )


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """Export results as PDF report"""
    if not REPORTLAB_AVAILABLE:
        return jsonify({
            'error': 'PDF export requires reportlab. Please install it with: pip install reportlab'
        }), 500
    
    from flask import Response
    import io
    from datetime import datetime
    
    data = request.json
    if not data or 'results' not in data or 'statistics' not in data:
        return jsonify({'error': 'No data to export'}), 400
    
    results = data['results']
    stats = data['statistics']
    unit = stats.get('unit', '')
    unit_label = f" {unit}" if unit else ""
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("Elevation Comparison Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report date
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary Statistics
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    stats_data = [
        ['Metric', 'Value'],
        ['Total Points', str(stats.get('total_points', 0))],
        ['Valid Points', str(stats.get('valid_points', 0))],
    ]
    
    if stats.get('outside_coverage', 0) > 0:
        stats_data.append(['Outside Coverage', str(stats.get('outside_coverage', 0))])
    
    if stats.get('valid_points', 0) > 0:
        stats_data.extend([
            ['Mean Error', f"{stats.get('mean_error', 0):.4f}{unit_label}"],
            ['RMSE', f"{stats.get('rmse', 0):.4f}{unit_label}"],
            ['Std Deviation', f"{stats.get('std_error', 0):.4f}{unit_label}"],
            ['Max Error', f"{stats.get('max_error', 0):.4f}{unit_label}"],
            ['Min Error', f"{stats.get('min_error', 0):.4f}{unit_label}"],
        ])
    
    stats_table = Table(stats_data, colWidths=[2*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Statistics by Type
    if stats.get('by_type'):
        story.append(Paragraph("Statistics by GCP Type", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        type_data = [['Type', 'Count', 'Mean Error', 'RMSE', 'Max Error', 'Min Error']]
        for gcp_type, type_stats in stats['by_type'].items():
            type_label = gcp_type if gcp_type else 'Unknown'
            type_data.append([
                type_label,
                str(type_stats.get('count', 0)),
                f"{type_stats.get('mean_error', 0):.4f}{unit_label}",
                f"{type_stats.get('rmse', 0):.4f}{unit_label}",
                f"{type_stats.get('max_error', 0):.4f}{unit_label}",
                f"{type_stats.get('min_error', 0):.4f}{unit_label}",
            ])
        
        type_table = Table(type_data, colWidths=[1.2*inch, 0.6*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        type_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(type_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Detailed Results Table
    story.append(Paragraph("Detailed Results", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    # Prepare table data
    table_data = [['GCP ID', 'Easting', 'Northing', f'GCP Elev{unit_label}', f'TIN Elev{unit_label}', 
                   f'Discrepancy{unit_label}', f'Abs Error{unit_label}', 'Type', 'Status']]
    
    for row in results:
        gcp_elev = f"{row.get('gcp_elevation', 0):.4f}" if row.get('gcp_elevation') is not None else 'N/A'
        tin_elev = f"{row.get('tin_elevation', 0):.4f}" if row.get('tin_elevation') is not None else 'N/A'
        discrepancy = f"{row.get('discrepancy', 0):.4f}" if row.get('discrepancy') is not None else 'N/A'
        abs_error = f"{row.get('absolute_error', 0):.4f}" if row.get('absolute_error') is not None else 'N/A'
        
        table_data.append([
            str(row.get('gcp_id', '')),
            f"{row.get('easting', 0):.2f}",
            f"{row.get('northing', 0):.2f}",
            gcp_elev,
            tin_elev,
            discrepancy,
            abs_error,
            str(row.get('gcp_type', '')),
            'OK' if row.get('status') == 'success' else 'Outside'
        ])
    
    # Create table with smaller font for detailed results
    results_table = Table(table_data, colWidths=[0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 
                                                  0.8*inch, 0.8*inch, 0.8*inch, 0.6*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(results_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return Response(
        buffer.getvalue(),
        mimetype='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=elevation_comparison_report.pdf'}
    )


if __name__ == '__main__':
    # Use PORT from environment (Cloud Run) or default to 5000 (local)
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)

