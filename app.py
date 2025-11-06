import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

app = Flask(__name__)

# إعدادات CORS للسماح لجميع المصادر (لتسهيل التطوير)
CORS(app, origins=[
    "https://petroai-web.web.app",
    "https://petroai-web.firebaseapp.com",
    "https://petroai-iq.web.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8080",
    "https://petroai-iq.web.app/plotly.html"
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
        # إنشاء figure أساسي
        fig = go.Figure()

        # إضافة جميع traces
        for trace in self.traces:
            fig.add_trace(trace)

        # إعداد تخطيط المحاور
        layout_updates = {
            'title': self.title,
            'template': self.template,
            'width': width,
            'height': height,
            'showlegend': True,
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
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
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'zeroline': False
            }

        # إعداد محور X
        layout_updates['xaxis'] = {
            'showgrid': True,
            'gridcolor': 'rgba(128,128,128,0.2)',
            'zeroline': False
        }

        fig.update_layout(**layout_updates)
        return fig


def allowed_file(filename):
    """التحقق من نوع الملف"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


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
        'service': 'PolyY Plotting API',
        'version': '2.0',
        'endpoints': {
            'upload': '/upload (POST) - Upload data file',
            'create_plot': '/create_plot (POST) - Create plot from JSON',
            'create_plot_from_file': '/create_plot_from_file (POST) - Create plot directly from file',
            'example_data': '/example_data (GET) - Get sample data'
        },
        'supported_formats': ['CSV', 'TXT', 'Excel (XLSX, XLS)'],
        'supported_plot_types': ['line', 'scatter', 'area', 'bar']
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

        # تنظيف البيانات - معالجة القيم الناقصة
        df_clean = df.copy()
        numeric_columns = df_clean.select_dtypes(
            include=[np.number]).columns.tolist()

        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # تحليل البيانات
        all_columns = df_clean.columns.tolist()

        # معاينة البيانات (أول 10 صفوف)
        preview_data = []
        for _, row in df_clean.head(10).iterrows():
            row_data = {}
            for col in all_columns:
                value = row[col]
                if pd.isna(value):
                    row_data[col] = None
                elif isinstance(value, (int, float)):
                    row_data[col] = float(value)
                else:
                    row_data[col] = str(value)
            preview_data.append(row_data)

        # إحصاءات الأعمدة الرقمية
        column_stats = {}
        for col in numeric_columns:
            if col in df_clean.columns:
                col_data = df_clean[col].dropna()
                if len(col_data) > 0:
                    column_stats[col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'count': int(len(col_data))
                    }

        response = {
            'success': True,
            'columns': all_columns,
            'numeric_columns': numeric_columns,
            'preview': preview_data,
            'total_rows': len(df_clean),
            'total_columns': len(all_columns),
            'column_stats': column_stats,
            'message': f'Successfully loaded {len(df_clean)} rows with {len(all_columns)} columns'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/create_plot', methods=['POST'])
def create_plot():
    """إنشاء مخطط PolyY من بيانات JSON"""
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

        # التحقق من صحة البيانات
        for i, trace in enumerate(traces_data):
            if 'x_data' not in trace or 'y_data' not in trace:
                return jsonify({'error': f'Trace {i+1} missing x_data or y_data'}), 400

            if len(trace['x_data']) != len(trace['y_data']):
                return jsonify({'error': f'Trace {i+1} has mismatched x and y data lengths'}), 400

        # إنشاء مخطط PolyY
        plotter = PolyYPlot(title=title, template=template)

        # إضافة جميع traces
        for trace_config in traces_data:
            plotter.add_trace(
                x_data=trace_config['x_data'],
                y_data=trace_config['y_data'],
                name=trace_config.get(
                    'name', f'Trace {len(plotter.traces) + 1}'),
                kind=trace_config.get('kind', 'line'),
                color=trace_config.get('color'),
                yaxis=trace_config.get('yaxis')
            )

        # إنشاء الشكل
        fig = plotter.create_figure(width=width, height=height)

        # تحويل إلى JSON للتفاعل
        plot_json = fig.to_json()

        response = {
            'success': True,
            # تحويل إلى dict لتفادي مشاكل التسلسل
            'plot_json': json.loads(plot_json),
            'traces_count': len(traces_data),
            'y_axes_count': len(plotter.y_axes),
            'title': title,
            'message': f'Successfully created plot with {len(traces_data)} traces and {len(plotter.y_axes)} Y-axes'
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

        if df.empty:
            return jsonify({'error': 'The uploaded file is empty'}), 400

        # الحصول على إعدادات الرسم من form data
        title = request.form.get('title', 'PolyY Chart')
        template = request.form.get('template', 'plotly')
        x_column = request.form.get('x_column')
        y_columns = request.form.getlist('y_columns[]')
        kinds = request.form.getlist('kinds[]')
        colors = request.form.getlist('colors[]')
        names = request.form.getlist('names[]')

        if not x_column:
            return jsonify({'error': 'X column is required'}), 400

        if not y_columns:
            return jsonify({'error': 'At least one Y column is required'}), 400

        if x_column not in df.columns:
            return jsonify({'error': f'X column "{x_column}" not found in data'}), 400

        # إنشاء مخطط PolyY
        plotter = PolyYPlot(title=title, template=template)

        # إضافة traces
        valid_traces = 0
        x_data = df[x_column].tolist()

        for i, y_col in enumerate(y_columns):
            if y_col and y_col in df.columns:
                y_data = pd.to_numeric(
                    df[y_col], errors='coerce').dropna().tolist()

                if len(y_data) > 0:
                    plotter.add_trace(
                        x_data=x_data[:len(y_data)],  # تأكد من تطابق الطول
                        y_data=y_data,
                        name=names[i] if i < len(
                            names) and names[i] else y_col,
                        kind=kinds[i] if i < len(
                            kinds) and kinds[i] else 'line',
                        color=colors[i] if i < len(
                            colors) and colors[i] else None
                    )
                    valid_traces += 1

        if valid_traces == 0:
            return jsonify({'error': 'No valid numeric data found in the specified Y columns'}), 400

        # إنشاء الشكل
        fig = plotter.create_figure()
        plot_json = fig.to_json()

        response = {
            'success': True,
            'plot_json': json.loads(plot_json),
            'traces_count': valid_traces,
            'x_column': x_column,
            'y_columns': [y_col for y_col in y_columns if y_col in df.columns],
            'message': f'Successfully created plot from file with {valid_traces} traces'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Error creating plot from file: {str(e)}'}), 500


@app.route('/example_data', methods=['GET'])
def get_example_data():
    """إرجاع بيانات مثاليه للاختبار"""
    # إنشاء بيانات مثاليه أكثر واقعية
    np.random.seed(42)  # للحصول على نتائج ثابتة

    timestamps = pd.date_range(
        '2024-01-01', periods=100, freq='H').strftime('%Y-%m-%d %H:%M:%S').tolist()

    # بيانات طاقة أكثر واقعية مع بعض الاتجاهات
    time_index = np.arange(100)

    example_data = {
        'timestamp': timestamps,
        'power_kwh': (50 + 10 * np.sin(time_index * 0.1) + np.random.normal(0, 3, 100)).tolist(),
        'voltage_v': (220 + 5 * np.cos(time_index * 0.05) + np.random.normal(0, 1, 100)).tolist(),
        'current_a': (15 + 3 * np.sin(time_index * 0.08) + np.random.normal(0, 0.5, 100)).tolist(),
        'temperature_c': (25 + 2 * np.sin(time_index * 0.03) + np.random.normal(0, 0.3, 100)).tolist(),
        'reactive_power_kvar': (10 + 2 * np.cos(time_index * 0.06) + np.random.normal(0, 0.4, 100)).tolist(),
        'efficiency': (0.85 + 0.1 * np.sin(time_index * 0.04) + np.random.normal(0, 0.02, 100)).tolist()
    }

    return jsonify({
        'success': True,
        'data': example_data,
        'description': 'Sample energy consumption data with 100 time points',
        'columns': {
            'timestamp': 'Time stamps',
            'power_kwh': 'Power consumption in kWh',
            'voltage_v': 'Voltage in volts',
            'current_a': 'Current in amperes',
            'temperature_c': 'Temperature in Celsius',
            'reactive_power_kvar': 'Reactive power in kVAR',
            'efficiency': 'System efficiency ratio'
        }
    })


@app.route('/test_plot', methods=['GET'])
def test_plot():
    """إنشاء رسم تجريبي للاختبار"""
    try:
        # بيانات تجريبية
        x_data = list(range(1, 101))

        plotter = PolyYPlot(title="Test PolyY Plot", template="plotly_dark")

        # إضافة عدة traces بأنماط مختلفة
        plotter.add_trace(
            x_data=x_data,
            y_data=[i + np.random.normal(0, 2) for i in x_data],
            name="Linear Trend",
            kind="line",
            color="#FF6B6B"
        )

        plotter.add_trace(
            x_data=x_data,
            y_data=[50 * np.sin(i * 0.1) + np.random.normal(0, 5)
                    for i in x_data],
            name="Sine Wave",
            kind="scatter",
            color="#4ECDC4"
        )

        plotter.add_trace(
            x_data=x_data,
            y_data=[i ** 0.5 * 10 + np.random.normal(0, 3) for i in x_data],
            name="Square Root",
            kind="area",
            color="#45B7D1"
        )

        fig = plotter.create_figure()
        plot_json = fig.to_json()

        return jsonify({
            'success': True,
            'plot_json': json.loads(plot_json),
            'message': 'Test plot generated successfully'
        })

    except Exception as e:
        return jsonify({'error': f'Error generating test plot: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
