import os
import base64
import io
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

app = Flask(__name__)

# إعدادات CORS
CORS(app, origins=[
    "https://petroai-web.web.app",
    "https://petroai-web.firebaseapp.com", 
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "https://petroai-iq.web.app"
])

# إعدادات التطبيق
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

class PolyYPlot:
    """فئة PolyY الرئيسية لإنشاء مخططات متعددة المحاور"""
    
    def __init__(self, title="PolyY Chart", template="plotly"):
        self.title = title
        self.template = template
        self.traces = []
        self.y_axes = []
        self.current_yaxis = 1
        
    def add_trace(self, x_data, y_data, name, kind="line", color=None, yaxis=None):
        """إضافة trace جديد إلى الرسم"""
        if yaxis is None:
            yaxis = f"y{self.current_yaxis}"
            self.current_yaxis += 1
        
        # إنشاء trace بناءً على النوع
        if kind == "line":
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                name=name,
                line=dict(color=color),
                yaxis=yaxis
            )
        elif kind == "scatter":
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                name=name,
                mode='markers',
                marker=dict(color=color),
                yaxis=yaxis
            )
        elif kind == "area":
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                name=name,
                fill='tozeroy',
                line=dict(color=color),
                yaxis=yaxis
            )
        elif kind == "bar":
            trace = go.Bar(
                x=x_data,
                y=y_data,
                name=name,
                marker=dict(color=color),
                yaxis=yaxis
            )
        else:
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                name=name,
                line=dict(color=color),
                yaxis=yaxis
            )
        
        self.traces.append(trace)
        
        # إعداد محور Y إذا كان جديداً
        if yaxis not in self.y_axes:
            self.y_axes.append(yaxis)
    
    def create_figure(self, width=1200, height=600):
        """إنشاء الشكل النهائي"""
        # إنشاء subplot مع محاور Y متعددة
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # إضافة جميع traces
        for trace in self.traces:
            fig.add_trace(trace)
        
        # إعداد تخطيط المحاور
        layout_updates = {
            'title': self.title,
            'template': self.template,
            'width': width,
            'height': height,
            'showlegend': True
        }
        
        # إعداد محاور Y المتعددة
        for i, yaxis in enumerate(self.y_axes):
            side = 'right' if i % 2 == 1 else 'left'
            position = 1.0 - (i * 0.15) if side == 'right' else None
            
            layout_updates[f'yaxis{i+1}'] = {
                'title': f'Y{i+1}',
                'side': side,
                'position': position,
                'overlaying': 'y' if i > 0 else None,
                'showgrid': False
            }
        
        fig.update_layout(**layout_updates)
        return fig

def allowed_file(filename):
    """التحقق من نوع الملف"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_data_file(file):
    """قراءة ملف البيانات بدعم للتنسيقات المختلفة"""
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(file)
        elif filename.endswith('.txt'):
            # محاولة قراءة ملف نصي بفاصل تبويب أو فاصلة
            try:
                return pd.read_csv(file, sep='\t')
            except:
                return pd.read_csv(file, sep=',')
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        'status': 'healthy',
        'service': 'PolyY Plotting Service',
        'version': '1.0',
        'features': 'Multi Y-axis plots, CSV/Excel support, Interactive charts'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """رفع ملف البيانات وتحليله"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV, TXT, or Excel files.'}), 400

        # قراءة البيانات
        df = read_data_file(file)
        
        if df.empty:
            return jsonify({'error': 'The uploaded file is empty'}), 400

        # تحليل البيانات
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        all_columns = df.columns.tolist()
        
        # معاينة البيانات
        preview_data = []
        for _, row in df.head(10).iterrows():
            preview_data.append({col: (float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])) 
                               for col in all_columns})

        response = {
            'success': True,
            'columns': all_columns,
            'numeric_columns': numeric_columns,
            'preview': preview_data,
            'total_rows': len(df),
            'total_columns': len(all_columns)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/create_plot', methods=['POST'])
def create_plot():
    """إنشاء مخطط PolyY"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # استخراج المعاملات
        title = data.get('title', 'PolyY Chart')
        template = data.get('template', 'plotly')
        width = data.get('width', 1200)
        height = data.get('height', 600)
        traces_data = data.get('traces', [])
        
        if not traces_data:
            return jsonify({'error': 'No traces data provided'}), 400

        # إنشاء مخطط PolyY
        plotter = PolyYPlot(title=title, template=template)
        
        # إضافة جميع traces
        for trace_config in traces_data:
            plotter.add_trace(
                x_data=trace_config['x_data'],
                y_data=trace_config['y_data'],
                name=trace_config['name'],
                kind=trace_config.get('kind', 'line'),
                color=trace_config.get('color'),
                yaxis=trace_config.get('yaxis')
            )
        
        # إنشاء الشكل
        fig = plotter.create_figure(width=width, height=height)
        
        # تحويل إلى HTML
        plot_html = pio.to_html(fig, include_plotlyjs='cdn', auto_open=False)
        
        # أو تحويل إلى JSON للتفاعل
        plot_json = fig.to_json()
        
        response = {
            'success': True,
            'plot_html': plot_html,
            'plot_json': plot_json,
            'traces_count': len(traces_data),
            'y_axes_count': len(plotter.y_axes)
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error creating plot: {str(e)}'}), 500

@app.route('/create_plot_from_file', methods=['POST'])
def create_plot_from_file():
    """إنشاء مخطط مباشرة من ملف بيانات"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # قراءة البيانات
        df = read_data_file(file)
        
        # الحصول على إعدادات الرسم من form data
        title = request.form.get('title', 'PolyY Chart')
        template = request.form.get('template', 'plotly')
        x_column = request.form.get('x_column')
        y_columns = request.form.getlist('y_columns[]')
        kinds = request.form.getlist('kinds[]')
        colors = request.form.getlist('colors[]')
        names = request.form.getlist('names[]')
        
        if not x_column or not y_columns:
            return jsonify({'error': 'X column and at least one Y column required'}), 400

        # إنشاء مخطط PolyY
        plotter = PolyYPlot(title=title, template=template)
        
        # إضافة traces
        x_data = df[x_column].tolist()
        
        for i, y_col in enumerate(y_columns):
            if y_col in df.columns:
                plotter.add_trace(
                    x_data=x_data,
                    y_data=df[y_col].tolist(),
                    name=names[i] if i < len(names) else y_col,
                    kind=kinds[i] if i < len(kinds) else 'line',
                    color=colors[i] if i < len(colors) else None
                )
        
        # إنشاء الشكل
        fig = plotter.create_figure()
        plot_html = pio.to_html(fig, include_plotlyjs='cdn', auto_open=False)
        
        response = {
            'success': True,
            'plot_html': plot_html,
            'traces_count': len(y_columns),
            'x_column': x_column,
            'y_columns': y_columns
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error creating plot from file: {str(e)}'}), 500

@app.route('/example_data', methods=['GET'])
def get_example_data():
    """إرجاع بيانات مثاليه للاختبار"""
    # بيانات مثاليه - استهلاك الطاقة
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
    
    example_data = {
        'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'power_kwh': np.random.normal(50, 10, 100).tolist(),
        'voltage_v': np.random.normal(220, 5, 100).tolist(),
        'current_a': np.random.normal(15, 3, 100).tolist(),
        'temperature_c': np.random.normal(25, 2, 100).tolist(),
        'reactive_power_kvar': np.random.normal(10, 2, 100).tolist()
    }
    
    return jsonify(example_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)