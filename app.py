import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد مكتبات التنبؤ
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    FORECAST_AVAILABLE = True
except ImportError:
    FORECAST_AVAILABLE = False

app = Flask(__name__)

# 🔧 إعدادات CORS المبسطة - إزالة التكرار
CORS(app, resources={
    r"/*": {
        "origins": [
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
            "http://127.0.0.1:5500"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    }
})

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
                line=dict(color=color, width=2),
                yaxis=yaxis
            )
        elif kind == "scatter":
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                name=name,
                mode='markers',
                marker=dict(color=color, size=6),
                yaxis=yaxis
            )
        elif kind == "area":
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                name=name,
                fill='tozeroy',
                line=dict(color=color, width=2),
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
                line=dict(color=color, width=2),
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
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': 'white' if self.template == 'plotly_dark' else 'black'},
            'margin': {'t': 50, 'r': 50, 'b': 80, 'l': 80}
        }

        # إعداد محاور Y المتعددة
        for i, yaxis in enumerate(self.y_axes):
            side = 'right' if i % 2 == 1 else 'left'
            position = 0.98 - (i * 0.15) if side == 'right' else 0.02

            layout_updates[f'yaxis{i+1}'] = {
                'title': f'Y{i+1}',
                'side': side,
                'position': position,
                'overlaying': 'y' if i > 0 else None,
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'zeroline': False,
                'showline': True,
                'linecolor': 'rgba(128,128,128,0.5)'
            }

        # إعداد محور X
        layout_updates['xaxis'] = {
            'showgrid': True,
            'gridcolor': 'rgba(128,128,128,0.2)',
            'zeroline': False,
            'showline': True,
            'linecolor': 'rgba(128,128,128,0.5)'
        }

        fig.update_layout(**layout_updates)
        return fig


class AdvancedForecaster:
    """فئة تنبؤ متقدمة مع تحليل محور X وتحسينات"""
    
    def __init__(self, lookback=10):
        self.lookback = lookback
        
    def prepare_dataframe(self, data_dict):
        """تحضير DataFrame من البيانات الواردة"""
        try:
            # إذا كانت البيانات قائمة من القواميس
            if isinstance(data_dict, list):
                return pd.DataFrame(data_dict)
            # إذا كانت البيانات قاموساً من القوائم
            elif isinstance(data_dict, dict):
                # التحقق إذا كانت القيم قوائم
                if all(isinstance(v, list) for v in data_dict.values()):
                    return pd.DataFrame(data_dict)
                else:
                    # إذا كانت البيانات بشكل آخر، حاول تحويلها
                    return pd.DataFrame([data_dict])
            else:
                raise ValueError("Unsupported data format")
        except Exception as e:
            raise ValueError(f"Error preparing dataframe: {str(e)}")
    
    def validate_data(self, data_dict, x_col, y_cols):
        """التحقق من صحة البيانات المدخلة"""
        try:
            # تحويل البيانات إلى DataFrame
            df = self.prepare_dataframe(data_dict)
            
            if df.empty:
                return False, "Empty dataset provided"
            
            if x_col not in df.columns:
                return False, f"X column '{x_col}' not found in data"
            
            # التحقق من وجود أعمدة Y المطلوبة
            missing_y_cols = [y_col for y_col in y_cols if y_col not in df.columns]
            if missing_y_cols:
                return False, f"Y columns not found in data: {missing_y_cols}"
            
            # التحقق من وجود بيانات كافية
            if len(df) < 10:
                return False, "Insufficient data for forecasting (minimum 10 records required)"
            
            return True, df
        
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
    
    def analyze_x_axis(self, x_data):
        """تحليل محور X لتحديد نوع البيانات والفترات"""
        try:
            # محاولة تحويل إلى تاريخ/وقت
            try:
                x_dates = pd.to_datetime(x_data)
                is_datetime = True
                # حساب الفترات بين التواريخ
                if len(x_dates) > 1:
                    time_diffs = [(x_dates[i] - x_dates[i-1]).total_seconds() / 3600 for i in range(1, len(x_dates))]
                    avg_interval_hours = np.mean(time_diffs)
                    return {
                        'type': 'datetime',
                        'values': x_dates,
                        'avg_interval_hours': avg_interval_hours,
                        'is_regular': np.std(time_diffs) < avg_interval_hours * 0.1  # فحص الانتظام
                    }
            except:
                pass
            
            # محاولة تحويل إلى أرقام
            try:
                x_numeric = pd.to_numeric(x_data, errors='coerce')
                if not x_numeric.isna().all():
                    x_numeric_clean = x_numeric.dropna()
                    if len(x_numeric_clean) > 1:
                        diffs = [x_numeric_clean.iloc[i] - x_numeric_clean.iloc[i-1] for i in range(1, len(x_numeric_clean))]
                        avg_interval = np.mean(diffs)
                        return {
                            'type': 'numeric',
                            'values': x_numeric_clean,
                            'avg_interval': avg_interval,
                            'is_regular': np.std(diffs) < abs(avg_interval) * 0.1
                        }
            except:
                pass
            
            # إذا فشل كل شيء، استخدم الفهرس
            return {
                'type': 'index',
                'values': pd.Series(range(len(x_data))),
                'avg_interval': 1,
                'is_regular': True
            }
            
        except Exception as e:
            print(f"Error analyzing X axis: {e}")
            return {
                'type': 'index',
                'values': pd.Series(range(len(x_data))),
                'avg_interval': 1,
                'is_regular': True
            }
    
    def generate_future_x(self, x_analysis, forecast_steps):
        """إنشاء قيم X مستقبلية بناءً على تحليل محور X"""
        last_x = x_analysis['values'].iloc[-1]
        
        if x_analysis['type'] == 'datetime':
            # إنشاء تواريخ مستقبلية
            interval_hours = x_analysis['avg_interval_hours']
            future_dates = [last_x + pd.Timedelta(hours=interval_hours * (i+1)) for i in range(forecast_steps)]
            return [date.strftime('%Y-%m-%d %H:%M:%S') for date in future_dates]
        elif x_analysis['type'] == 'numeric':
            # إنشاء قيم رقمية مستقبلية
            interval = x_analysis['avg_interval']
            future_values = [float(last_x + interval * (i+1)) for i in range(forecast_steps)]
            return future_values
        else:
            # استخدام الفهرس
            last_index = int(last_x) if isinstance(last_x, (int, float)) else len(x_analysis['values']) - 1
            future_indices = [last_index + i + 1 for i in range(forecast_steps)]
            return future_indices
    
    def advanced_forecast_method(self, y_data, forecast_steps):
        """طريقة تنبؤ متقدمة باستخدام تحليل الاتجاه والأنماط"""
        if len(y_data) < 10:
            return [], []
        
        try:
            # تحليل الاتجاه باستخدام الانحدار الخطي
            x_trend = np.arange(len(y_data)).reshape(-1, 1)
            trend_model = LinearRegression()
            trend_model.fit(x_trend, y_data)
            trend_coef = trend_model.coef_[0]
            
            # حساب الموسمية (إن وجدت)
            seasonal_component = self.detect_seasonality(y_data)
            
            # حساب التباين
            volatility = np.std(y_data[-10:]) if len(y_data) >= 10 else np.std(y_data)
            
            # التنبؤ المستقبلي
            future_predictions = []
            last_value = y_data[-1]
            
            for i in range(forecast_steps):
                # الجمع بين الاتجاه والموسمية والضوضاء
                trend_part = trend_coef * (i + 1)
                seasonal_part = seasonal_component[i % len(seasonal_component)] if seasonal_component else 0
                noise_part = np.random.normal(0, volatility * 0.2)
                
                next_value = last_value + trend_part + seasonal_part + noise_part
                
                # التأكد من أن القيم واقعية
                if np.min(y_data) >= 0 and next_value < 0:
                    next_value = max(0, next_value)
                    
                future_predictions.append(float(next_value))
            
            # إنشاء تنبؤات تاريخية للمقارنة
            historical_fit = trend_model.predict(x_trend).tolist()
            
            return future_predictions, historical_fit
            
        except Exception as e:
            print(f"Advanced forecast error: {e}")
            # استخدام طريقة احتياطية
            return self.moving_average_forecast(y_data, forecast_steps)
    
    def detect_seasonality(self, data):
        """كشف الأنماط الموسمية في البيانات"""
        if len(data) < 20:
            return []
        
        try:
            # تحويل البيانات إلى سلسلة زمنية
            ts = pd.Series(data)
            
            # حساب الارتباط الذاتي للكشف عن الموسمية
            autocorr = []
            max_lag = min(10, len(data) // 4)
            
            for lag in range(1, max_lag + 1):
                if lag < len(data):
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    autocorr.append(corr)
            
            # إذا كان هناك ارتباط ذاتي قوي، يوجد نمط موسمي
            if autocorr and max(autocorr) > 0.5:
                best_lag = np.argmax(autocorr) + 1
                seasonal_pattern = data[-best_lag:] if len(data) >= best_lag else []
                return seasonal_pattern
            
            return []
        except:
            return []
    
    def moving_average_forecast(self, y_data, forecast_steps):
        """طريقة المتوسط المتحرك كبديل"""
        if len(y_data) < 5:
            return [], []
        
        window_size = min(5, len(y_data) // 4)
        moving_avg = []
        
        for i in range(window_size, len(y_data)):
            window = y_data[i-window_size:i]
            moving_avg.append(np.mean(window))
        
        if len(moving_avg) < 2:
            return [], []
        
        # التنبؤ البسيط
        last_avg = moving_avg[-1]
        last_trend = moving_avg[-1] - moving_avg[-2] if len(moving_avg) > 1 else 0
        volatility = np.std(y_data[-window_size:])
        
        future_predictions = []
        for i in range(forecast_steps):
            next_value = last_avg + last_trend * (i + 1) + np.random.normal(0, volatility * 0.3)
            future_predictions.append(float(next_value))
        
        return future_predictions, moving_avg
    
    def forecast(self, data_dict, x_col, y_cols, forecast_percentage=0.25):
        """التنبؤ بالقيم المستقبلية مع تحليل محور X"""
        try:
            # التحقق من صحة البيانات
            is_valid, validation_result = self.validate_data(data_dict, x_col, y_cols)
            if not is_valid:
                return {'success': False, 'error': validation_result}
            
            df = validation_result
            
            # تحليل محور X
            x_analysis = self.analyze_x_axis(df[x_col])
            print(f"X-axis analysis: {x_analysis['type']}, regular: {x_analysis.get('is_regular', False)}")
            
            # تحديد عدد خطوات التنبؤ (25% من البيانات)
            total_length = len(df)
            forecast_steps = max(3, int(total_length * forecast_percentage))
            print(f"Forecast steps: {forecast_steps} (25% of {total_length})")
            
            forecasts = {}
            future_x_values = {}
            historical_predictions = {}
            columns_forecasted = []
            
            for y_col in y_cols:
                if y_col not in df.columns:
                    print(f"Y column {y_col} not found in data")
                    continue
                    
                # استخراج البيانات وتنظيفها
                y_data_series = pd.to_numeric(df[y_col], errors='coerce')
                valid_indices = y_data_series.notna()
                
                valid_count = valid_indices.sum()
                print(f"Column {y_col}: {valid_count} valid values out of {len(df)}")
                
                if valid_count < 10:  # تقليل الحد الأدنى إلى 10
                    print(f"Column {y_col}: insufficient valid data ({valid_count} < 10)")
                    continue
                
                # استخدام البيانات الصالحة فقط
                y_data = y_data_series[valid_indices].values
                
                # إنشاء قيم X مستقبلية
                future_x = self.generate_future_x(x_analysis, forecast_steps)
                future_x_values[y_col] = future_x
                
                # استخدام طريقة التنبؤ المتقدمة
                future_predictions, historical_fit = self.advanced_forecast_method(y_data, forecast_steps)
                
                if not future_predictions:
                    print(f"Column {y_col}: no predictions generated")
                    continue
                
                forecasts[y_col] = future_predictions
                historical_predictions[y_col] = historical_fit
                columns_forecasted.append(y_col)
            
            if not forecasts:
                return {'success': False, 'error': 'No successful forecasts generated for any column'}
            
            return {
                'success': True,
                'forecasts': forecasts,
                'future_x_values': future_x_values,
                'historical_predictions': historical_predictions,
                'forecast_steps': forecast_steps,
                'lookback': self.lookback,
                'x_analysis': {
                    'type': x_analysis['type'],
                    'is_regular': x_analysis.get('is_regular', False)
                },
                'method': 'advanced_forecasting',
                'columns_forecasted': columns_forecasted
            }
            
        except Exception as e:
            print(f"Forecasting error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Forecasting error: {str(e)}'
            }


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
        raise ValueError(f"Error reading file: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        'status': 'healthy',
        'service': 'PolyY Plotting API',
        'version': '2.3',
        'endpoints': {
            'upload': '/upload (POST) - Upload data file',
            'create_plot': '/create_plot (POST) - Create plot from JSON',
            'create_plot_from_file': '/create_plot_from_file (POST) - Create plot directly from file',
            'forecast': '/forecast (POST) - Advanced Forecasting',
            'example_data': '/example_data (GET) - Get sample data'
        },
        'supported_formats': ['CSV', 'TXT', 'Excel (XLSX, XLS)'],
        'supported_plot_types': ['line', 'scatter', 'area', 'bar'],
        'features': {
            'forecasting_available': FORECAST_AVAILABLE,
            'multi_y_axis': True,
            'advanced_forecasting': True,
            'x_axis_analysis': True,
            'seasonality_detection': True
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
                name=trace_config.get('name', f'Trace {len(plotter.traces) + 1}'),
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
                y_data = pd.to_numeric(df[y_col], errors='coerce').dropna().tolist()

                if len(y_data) > 0:
                    plotter.add_trace(
                        x_data=x_data[:len(y_data)],  # تأكد من تطابق الطول
                        y_data=y_data,
                        name=names[i] if i < len(names) and names[i] else y_col,
                        kind=kinds[i] if i < len(kinds) and kinds[i] else 'line',
                        color=colors[i] if i < len(colors) and colors[i] else None
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


@app.route('/forecast', methods=['POST'])
def forecast():
    """التنبؤ المتقدم مع تحليل محور X"""
    try:
        if not FORECAST_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Forecasting not available. scikit-learn is required.'
            }), 500

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # استخراج البيانات مع قيم افتراضية
        data_dict = data.get('data', {})
        x_col = data.get('x_column')
        y_cols = data.get('y_columns', [])
        chart_type = data.get('chart_type', 'line')
        forecast_percentage = data.get('forecast_percentage', 0.25)

        print(f"Received forecast request: x_col={x_col}, y_cols={y_cols}, chart_type={chart_type}")
        print(f"Data type: {type(data_dict)}, Data keys: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'list of dicts'}")

        if not data_dict:
            return jsonify({'error': 'No data provided for forecasting'}), 400

        if not x_col:
            return jsonify({'error': 'X column is required'}), 400

        if not y_cols:
            return jsonify({'error': 'At least one Y column is required'}), 400

        # التحقق من أن نوع الرسم هو line chart
        if chart_type != 'line':
            return jsonify({'error': 'Forecasting is only available for line charts'}), 400

        # التحقق من أن البيانات تحتوي على الأعمدة المطلوبة
        if isinstance(data_dict, list) and len(data_dict) > 0:
            # بيانات بشكل قائمة من القواميس
            if x_col not in data_dict[0]:
                return jsonify({'error': f'X column "{x_col}" not found in data'}), 400
        elif isinstance(data_dict, dict):
            # بيانات بشكل قاموس من القوائم
            if x_col not in data_dict:
                return jsonify({'error': f'X column "{x_col}" not found in data'}), 400
        else:
            return jsonify({'error': 'Invalid data format'}), 400

        # إنشاء وتنفيذ التنبؤ المتقدم
        forecaster = AdvancedForecaster(lookback=10)
        result = forecaster.forecast(data_dict, x_col, y_cols, forecast_percentage)

        if not result['success']:
            return jsonify({'error': result['error']}), 400

        response = {
            'success': True,
            'forecasts': result['forecasts'],
            'future_x_values': result['future_x_values'],
            'historical_predictions': result['historical_predictions'],
            'forecast_steps': result['forecast_steps'],
            'lookback': result['lookback'],
            'x_analysis': result['x_analysis'],
            'method': result['method'],
            'columns_forecasted': result.get('columns_forecasted', list(result['forecasts'].keys())),
            'message': f'Successfully generated advanced forecasts for {len(result["forecasts"])} columns'
        }

        return jsonify(response)

    except Exception as e:
        print(f"Forecasting error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating forecasts: {str(e)}'}), 500


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
            y_data=[50 * np.sin(i * 0.1) + np.random.normal(0, 5) for i in x_data],
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
