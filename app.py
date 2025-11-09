import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# محاولة استيراد TensorFlow بشكل آمن
TENSORFLOW_AVAILABLE = False
LSTM_FORECASTING_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    LSTM_FORECASTING_AVAILABLE = True
    logger.info("✅ TensorFlow and scikit-learn imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ TensorFlow/scikit-learn not available: {e}")
    # استخدام بدائل بسيطة إذا لم يتوفر TensorFlow
    try:
        from sklearn.preprocessing import MinMaxScaler
        logger.info("✅ scikit-learn available for basic scaling")
    except ImportError:
        logger.warning("⚠️ scikit-learn also not available")

app = Flask(__name__)

# 🔧 إعدادات CORS الموسعة للسماح لجميع النطاقات
CORS(app, origins=[
    "https://petroai-web.web.app",
    "https://petroai-web.firebaseapp.com", 
    "https://petroai-iq.web.app",
    "https://petroai-iq.firebaseapp.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://*.web.app",
    "https://*.firebaseapp.com"
])

# إعدادات التطبيق
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# إضافة headers يدوياً لجميع الردود
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '86400')  # 24 hours
    return response

# معالجة طلبات OPTIONS (preflight) يدوياً
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "success"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

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
        try:
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
                
            return True
        except Exception as e:
            logger.error(f"Error adding trace: {e}")
            return False

    def create_figure(self, width=1200, height=600):
        """إنشاء الشكل النهائي"""
        try:
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
        except Exception as e:
            logger.error(f"Error creating figure: {e}")
            return None


class LSTMForecaster:
    """فئة للتنبؤ باستخدام LSTM"""
    
    def __init__(self, lookback=10, forecast_steps=10):
        self.lookback = lookback
        self.forecast_steps = forecast_steps
        self.scalers = {}
        self.models = {}
        
    def create_sequences(self, data, lookback):
        """إنشاء متواليات للتدريب"""
        try:
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape):
        """بناء نموذج LSTM"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return None
    
    def forecast(self, data_dict, x_col, y_cols, forecast_percentage=0.25):
        """التنبؤ بالقيم المستقبلية"""
        try:
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(data_dict)
            
            if df.empty:
                return {'success': False, 'error': 'Empty dataset provided'}
            
            # تحديد عدد خطوات التنبؤ (25% من البيانات)
            total_length = len(df)
            forecast_steps = max(3, int(total_length * forecast_percentage))
            
            forecasts = {}
            historical_predictions = {}
            
            for y_col in y_cols:
                if y_col not in df.columns:
                    logger.warning(f"Column {y_col} not found in dataset")
                    continue
                    
                # استخراج البيانات وتنظيفها
                y_data = pd.to_numeric(df[y_col], errors='coerce').dropna().values
                
                if len(y_data) < self.lookback + 5:
                    logger.warning(f"Insufficient data for {y_col}: {len(y_data)} points, need at least {self.lookback + 5}")
                    continue
                
                # تطبيع البيانات
                try:
                    scaler = MinMaxScaler()
                    y_scaled = scaler.fit_transform(y_data.reshape(-1, 1)).flatten()
                except Exception as e:
                    logger.error(f"Error scaling data for {y_col}: {e}")
                    continue
                
                # إنشاء متواليات
                X, y = self.create_sequences(y_scaled, self.lookback)
                
                if len(X) == 0:
                    logger.warning(f"No valid sequences created for {y_col}")
                    continue
                
                # بناء وتدريب النموذج
                model = self.build_model((self.lookback, 1))
                if model is None:
                    continue
                
                try:
                    # تدريب سريع (للاستخدام الفوري)
                    model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
                    
                    # التنبؤ التاريخي (للتحقق من جودة النموذج)
                    historical_pred = model.predict(X, verbose=0)
                    historical_pred = scaler.inverse_transform(historical_pred).flatten()
                    
                    # التنبؤ المستقبلي
                    last_sequence = y_scaled[-self.lookback:].reshape(1, self.lookback, 1)
                    future_predictions = []
                    
                    current_sequence = last_sequence.copy()
                    for _ in range(forecast_steps):
                        next_pred = model.predict(current_sequence, verbose=0)
                        future_predictions.append(next_pred[0, 0])
                        # تحديث المتوالية بإضافة التنبؤ وإزالة القيمة الأولى
                        current_sequence = np.roll(current_sequence, -1, axis=1)
                        current_sequence[0, -1, 0] = next_pred[0, 0]
                    
                    future_predictions = scaler.inverse_transform(
                        np.array(future_predictions).reshape(-1, 1)
                    ).flatten()
                    
                    forecasts[y_col] = future_predictions.tolist()
                    historical_predictions[y_col] = historical_pred.tolist()
                    self.scalers[y_col] = scaler
                    self.models[y_col] = model
                    
                    logger.info(f"✅ Successfully forecasted {y_col} - {forecast_steps} steps")
                    
                except Exception as e:
                    logger.error(f"Error in model training/prediction for {y_col}: {e}")
                    continue
            
            if not forecasts:
                return {'success': False, 'error': 'No successful forecasts generated for any column'}
            
            return {
                'success': True,
                'forecasts': forecasts,
                'historical_predictions': historical_predictions,
                'forecast_steps': forecast_steps,
                'lookback': self.lookback
            }
            
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            return {
                'success': False,
                'error': f'Forecasting error: {str(e)}'
            }


class SimpleForecaster:
    """فئة تنبؤ بسيطة بدون TensorFlow"""
    
    def __init__(self, forecast_steps=10):
        self.forecast_steps = forecast_steps
        
    def simple_forecast(self, data_dict, x_col, y_cols, forecast_percentage=0.25):
        """تنبؤ بسيط باستخدام متوسط متحرك"""
        try:
            df = pd.DataFrame(data_dict)
            
            if df.empty:
                return {'success': False, 'error': 'Empty dataset provided'}
            
            total_length = len(df)
            forecast_steps = max(3, int(total_length * forecast_percentage))
            
            forecasts = {}
            
            for y_col in y_cols:
                if y_col not in df.columns:
                    continue
                    
                # استخراج البيانات وتنظيفها
                y_data = pd.to_numeric(df[y_col], errors='coerce').dropna().values
                
                if len(y_data) < 5:
                    continue
                
                # استخدام متوسط متحرك بسيط للتنبؤ
                window_size = min(5, len(y_data) // 2)
                if window_size < 2:
                    window_size = 2
                
                # حساب المتوسط المتحرك
                moving_avg = np.convolve(y_data, np.ones(window_size)/window_size, mode='valid')
                
                if len(moving_avg) < 2:
                    continue
                
                # التنبؤ البسيط باستخدام الاتجاه الأخير
                last_trend = moving_avg[-1] - moving_avg[-2] if len(moving_avg) > 1 else 0
                last_value = y_data[-1]
                
                future_predictions = []
                for i in range(forecast_steps):
                    # إضافة الاتجاه مع بعض العشوائية
                    next_value = last_value + last_trend * (1 + 0.1 * np.random.normal())
                    future_predictions.append(next_value)
                    last_value = next_value
                
                forecasts[y_col] = future_predictions
            
            if not forecasts:
                return {'success': False, 'error': 'No successful forecasts generated'}
            
            return {
                'success': True,
                'forecasts': forecasts,
                'forecast_steps': forecast_steps,
                'method': 'simple_moving_average',
                'message': 'Used simple moving average (TensorFlow not available)'
            }
            
        except Exception as e:
            logger.error(f"Simple forecasting error: {e}")
            return {'success': False, 'error': f'Simple forecasting error: {str(e)}'}


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
                file.seek(0)  # إعادة تعيين المؤشر
                return pd.read_csv(file, sep=',')
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        raise ValueError(f"Error reading file: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        'status': 'healthy',
        'service': 'PolyY Plotting API',
        'version': '2.1',
        'endpoints': {
            'upload': '/upload (POST) - Upload data file',
            'create_plot': '/create_plot (POST) - Create plot from JSON',
            'create_plot_from_file': '/create_plot_from_file (POST) - Create plot directly from file',
            'forecast': '/forecast (POST) - LSTM forecasting',
            'example_data': '/example_data (GET) - Get sample data'
        },
        'supported_formats': ['CSV', 'TXT', 'Excel (XLSX, XLS)'],
        'supported_plot_types': ['line', 'scatter', 'area', 'bar'],
        'features': {
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'lstm_forecasting': LSTM_FORECASTING_AVAILABLE,
            'multi_y_axis': True,
            'forecasting': True
        }
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
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

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

        logger.info(f"File uploaded successfully: {file.filename}, {len(df_clean)} rows")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing file: {e}")
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
        successful_traces = 0
        for trace_config in traces_data:
            success = plotter.add_trace(
                x_data=trace_config['x_data'],
                y_data=trace_config['y_data'],
                name=trace_config.get('name', f'Trace {successful_traces + 1}'),
                kind=trace_config.get('kind', 'line'),
                color=trace_config.get('color'),
                yaxis=trace_config.get('yaxis')
            )
            if success:
                successful_traces += 1

        if successful_traces == 0:
            return jsonify({'error': 'No valid traces could be created'}), 400

        # إنشاء الشكل
        fig = plotter.create_figure(width=width, height=height)
        
        if fig is None:
            return jsonify({'error': 'Failed to create plot figure'}), 500

        # تحويل إلى JSON للتفاعل
        plot_json = fig.to_json()

        response = {
            'success': True,
            'plot_json': json.loads(plot_json),
            'traces_count': successful_traces,
            'y_axes_count': len(plotter.y_axes),
            'title': title,
            'message': f'Successfully created plot with {successful_traces} traces and {len(plotter.y_axes)} Y-axes'
        }

        logger.info(f"Plot created successfully: {title}, {successful_traces} traces")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error creating plot: {e}")
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
                y_data = pd.to_numeric(df[y_col], errors='coerce').dropna().tolist()

                if len(y_data) > 0:
                    # تأكد من تطابق الطول بين X و Y
                    min_length = min(len(x_data), len(y_data))
                    success = plotter.add_trace(
                        x_data=x_data[:min_length],
                        y_data=y_data[:min_length],
                        name=names[i] if i < len(names) and names[i] else y_col,
                        kind=kinds[i] if i < len(kinds) and kinds[i] else 'line',
                        color=colors[i] if i < len(colors) and colors[i] else None
                    )
                    if success:
                        valid_traces += 1

        if valid_traces == 0:
            return jsonify({'error': 'No valid numeric data found in the specified Y columns'}), 400

        # إنشاء الشكل
        fig = plotter.create_figure()
        if fig is None:
            return jsonify({'error': 'Failed to create plot figure'}), 500
            
        plot_json = fig.to_json()

        response = {
            'success': True,
            'plot_json': json.loads(plot_json),
            'traces_count': valid_traces,
            'x_column': x_column,
            'y_columns': [y_col for y_col in y_columns if y_col in df.columns],
            'message': f'Successfully created plot from file with {valid_traces} traces'
        }

        logger.info(f"Plot from file created: {file.filename}, {valid_traces} traces")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error creating plot from file: {e}")
        return jsonify({'error': f'Error creating plot from file: {str(e)}'}), 500


@app.route('/forecast', methods=['POST'])
def forecast():
    """التنبؤ باستخدام LSTM أو طريقة بسيطة"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # استخراج البيانات
        data_dict = data.get('data', {})
        x_col = data.get('x_column')
        y_cols = data.get('y_columns', [])
        chart_type = data.get('chart_type', 'line')

        if not data_dict:
            return jsonify({'error': 'No data provided for forecasting'}), 400

        if not x_col:
            return jsonify({'error': 'X column is required'}), 400

        if not y_cols:
            return jsonify({'error': 'At least one Y column is required'}), 400

        # التحقق من أن نوع الرسم هو line chart
        if chart_type != 'line':
            return jsonify({'error': 'Forecasting is only available for line charts'}), 400

        result = None
        
        # استخدام LSTM إذا كان متاحاً
        if LSTM_FORECASTING_AVAILABLE:
            logger.info("Using LSTM for forecasting")
            forecaster = LSTMForecaster(lookback=10, forecast_steps=10)
            result = forecaster.forecast(data_dict, x_col, y_cols)
        else:
            # استخدام طريقة بسيطة
            logger.info("Using simple forecasting (LSTM not available)")
            simple_forecaster = SimpleForecaster(forecast_steps=10)
            result = simple_forecaster.simple_forecast(data_dict, x_col, y_cols)

        if not result['success']:
            return jsonify({'error': result['error']}), 400

        response = {
            'success': True,
            'forecasts': result['forecasts'],
            'forecast_steps': result['forecast_steps'],
            'lookback': result.get('lookback', 0),
            'method': result.get('method', 'lstm'),
            'message': result.get('message', f'Successfully generated forecasts for {len(result["forecasts"])} columns')
        }

        # إضافة التنبؤات التاريخية إذا كانت متاحة
        if 'historical_predictions' in result:
            response['historical_predictions'] = result['historical_predictions']

        logger.info(f"Forecast completed: {len(result['forecasts'])} columns, {result['forecast_steps']} steps")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error generating forecasts: {e}")
        return jsonify({'error': f'Error generating forecasts: {str(e)}'}), 500


@app.route('/example_data', methods=['GET'])
def get_example_data():
    """إرجاع بيانات مثاليه للاختبار"""
    try:
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
    except Exception as e:
        logger.error(f"Error generating example data: {e}")
        return jsonify({'error': f'Error generating example data: {str(e)}'}), 500


@app.route('/test_plot', methods=['GET'])
def test_plot():
    """إنشاء رسم تجريبي للاختبار"""
    try:
        # بيانات تجريبية
        x_data = list(range(1, 101))

        plotter = PolyYPlot(title="Test PolyY Plot", template="plotly_dark")

        # إضافة عدة traces بأنماط مختلفة
        traces_added = 0
        traces_added += 1 if plotter.add_trace(
            x_data=x_data,
            y_data=[i + np.random.normal(0, 2) for i in x_data],
            name="Linear Trend",
            kind="line",
            color="#FF6B6B"
        ) else 0

        traces_added += 1 if plotter.add_trace(
            x_data=x_data,
            y_data=[50 * np.sin(i * 0.1) + np.random.normal(0, 5) for i in x_data],
            name="Sine Wave",
            kind="scatter",
            color="#4ECDC4"
        ) else 0

        traces_added += 1 if plotter.add_trace(
            x_data=x_data,
            y_data=[i ** 0.5 * 10 + np.random.normal(0, 3) for i in x_data],
            name="Square Root",
            kind="area",
            color="#45B7D1"
        ) else 0

        if traces_added == 0:
            return jsonify({'error': 'Failed to create test traces'}), 500

        fig = plotter.create_figure()
        if fig is None:
            return jsonify({'error': 'Failed to create test plot figure'}), 500
            
        plot_json = fig.to_json()

        return jsonify({
            'success': True,
            'plot_json': json.loads(plot_json),
            'message': 'Test plot generated successfully'
        })

    except Exception as e:
        logger.error(f"Error generating test plot: {e}")
        return jsonify({'error': f'Error generating test plot: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting PolyY Plotting API on port {port}")
    logger.info(f"📊 TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    logger.info(f"🤖 LSTM Forecasting Available: {LSTM_FORECASTING_AVAILABLE}")
    logger.info(f"🔧 Debug Mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
