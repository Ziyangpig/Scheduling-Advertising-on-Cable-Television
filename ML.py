import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 传统时间序列方法
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

# 机器学习方法
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class TelevisionRatingsForecaster:
    """
    电视收视率预测器
    对应论文中 'Ratings Forecasts' 部分
    """
    
    def __init__(self, demographic_groups=16, quarter_hours=96):
        self.demographic_groups = demographic_groups
        self.quarter_hours = quarter_hours
        self.best_models = {}  # 存储每个序列的最佳模型
        self.performance_metrics = {}
        
    def generate_sample_data(self, periods=365*2):
        """
        生成模拟收视率数据
        基于论文中描述的数据特性：
        - 16个人口统计分组
        - 96个15分钟时段
        - 高变异性 (CV≈0.5)
        - 天和周季节性
        """
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=periods, freq='15T')
        
        data = []
        for demo in range(1, self.demographic_groups + 1):
            for period in range(periods):
                date = dates[period]
                
                # 基础模式 + 季节性 + 噪声
                hour = date.hour
                day_of_week = date.dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
                
                # 基础收视模式 (早晚高峰)
                base_pattern = (
                    10 * np.exp(-(hour - 8)**2 / 8) +  # 早晨高峰
                    15 * np.exp(-(hour - 19)**2 / 6)   # 晚间高峰
                )
                
                # 星期效应
                day_effect = 1.0
                if day_of_week == 0:  # 周一
                    day_effect = 1.1
                elif day_of_week == 4:  # 周五
                    day_effect = 1.15
                elif is_weekend:
                    day_effect = 1.3
                
                # 生成收视率 (千次观看)
                rating = base_pattern * day_effect * np.random.lognormal(0, 0.3)
                rating = max(rating, 0.1)  # 确保非负
                
                data.append({
                    'timestamp': date,
                    'demographic': f'Demo_{demo:02d}',
                    'quarter_hour': period % 96,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'rating': rating
                })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')  # 设置时间索引
        return df.reset_index()  # 返回时重置索引以保持列格式
    
    def create_features(self, df):
        """创建时间序列特征"""
        df = df.copy()
        
        # 确保时间戳列存在
        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df = df.reset_index()
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 滞后特征
        for lag in [1, 2, 3, 24, 168]:  # 1期, 2期, 3期, 1天, 1周
            df[f'rating_lag_{lag}'] = df.groupby('demographic')['rating'].shift(lag)
        
        # 滚动统计特征
        for window in [4, 12, 24]:  # 1小时, 3小时, 6小时
            df[f'rating_roll_mean_{window}'] = df.groupby('demographic')['rating'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rating_roll_std_{window}'] = df.groupby('demographic')['rating'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        return df
    
    def prepare_time_series_data(self, df, demographic):
        """为传统时间序列模型准备数据"""
        demo_data = df[df['demographic'] == demographic].copy()
        
        if len(demo_data) == 0:
            return None
            
        # 创建正确的时间索引序列
        demo_data = demo_data.set_index('timestamp')
        demo_data = demo_data.asfreq('15T')  # 确保15分钟频率
        
        # 处理缺失值
        demo_data['rating'] = demo_data['rating'].fillna(method='ffill')
        
        return demo_data['rating']
    
    def historical_average(self, train_data, test_dates, demographic, quarter_hour):
        """历史平均法 (基准模型)"""
        # 相同时段的历史平均
        train_subset = train_data[
            (train_data['demographic'] == demographic) & 
            (train_data['quarter_hour'] == quarter_hour)
        ]
        return train_subset['rating'].mean() if not train_subset.empty else 0.1
    
    def holt_winters_forecast(self, train_series, forecast_horizon=1):
        """Holt-Winters 三指数平滑 - 修复季节性周期问题"""
        print('holt_winters_forecast')
        try:
            # 检查数据长度是否足够 (至少2个完整季节周期)
            seasonal_periods = 672  # 一周的15分钟时段数 (96*7)
            min_required_length = 2 * seasonal_periods  # 至少2周数据
            
            if len(train_series) < min_required_length:
                # 数据不足时使用简化版本或回退方法
                if len(train_series) >= 96:  # 至少一天数据
                    # 使用日季节性而非周季节性
                    daily_seasonal_periods = 96  # 一天的时段数
                    model = ExponentialSmoothing(
                        train_series,
                        seasonal_periods=daily_seasonal_periods,
                        trend='add',
                        seasonal='mul',
                        initialization_method='estimated'
                    )
                    fitted_model = model.fit()
                else:
                    # 数据太少，使用简单指数平滑
                    model = ExponentialSmoothing(
                        train_series,
                        trend='add',
                        seasonal=None,  # 无季节性
                        initialization_method='estimated'
                    )
                    fitted_model = model.fit()
            else:
                # 数据充足，使用完整的周季节性模型
                model = ExponentialSmoothing(
                    train_series,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='mul',
                    initialization_method='estimated'
                )
                fitted_model = model.fit()
            
            forecast = fitted_model.forecast(forecast_horizon)
            return forecast.values[0] if forecast_horizon == 1 else forecast.values
            
        except Exception as e:
            print(f"Holt-Winters错误: {e}")
            # 回退到加权移动平均
            if len(train_series) >= 24:
                # 使用指数加权移动平均
                weights = np.exp(np.linspace(-1, 0, min(24, len(train_series))))
                weights /= weights.sum()
                recent_data = train_series[-len(weights):]
                return np.dot(recent_data, weights)
            else:
                return np.mean(train_series[-24:])  # 简单近期平均
    
    def simplified_holt_winters(self, train_series, forecast_horizon=1, seasonal_periods=96):
        """简化版Holt-Winters，适应数据长度"""
        print('simplified_holt_winters')
        try:
            # 确保数据长度足够
            if len(train_series) < seasonal_periods * 2:
                seasonal_periods = min(seasonal_periods, len(train_series) // 2)
            
            if seasonal_periods < 2:
                # 数据太少，使用简单指数平滑
                model = ExponentialSmoothing(
                    train_series,
                    trend='add',
                    seasonal=None,
                    initialization_method='heuristic'
                )
            else:
                model = ExponentialSmoothing(
                    train_series,
                    seasonal_periods=seasonal_periods,
                    trend='add',
                    seasonal='mul',
                    initialization_method='heuristic'
                )
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(forecast_horizon)
            return forecast.values[0] if forecast_horizon == 1 else forecast.values
            
        except Exception as e:
            print(f"简化Holt-Winters错误: {e}")
            return np.mean(train_series[-24:])
    
    def robust_arima_forecast(self, train_series, max_order=(2,1,2), seasonal_order=(1,1,1,96)):
        """稳健的ARIMA预测 - 处理收敛问题"""
        print('robust_arima_forecast')
        try:
            # 根据数据长度智能选择ARIMA参数
            n = len(train_series)
            
            # 简化参数以避免收敛问题
            if n < 100:
                order = (1, 1, 1)  # 非常简单的模型
                use_seasonal = False
            elif n < 300:
                order = (1, 1, 1)
                use_seasonal = (1, 0, 1, 24) if n >= 200 else False  # 日季节性
            else:
                order = (2, 1, 2)
                use_seasonal = seasonal_order if n >= 2 * seasonal_order[3] else (1, 0, 1, 24)
            
            # 数据标准化以提高数值稳定性
            series_mean = train_series.mean()
            series_std = train_series.std()
            
            if series_std > 0:
                normalized_series = (train_series - series_mean) / series_std
            else:
                normalized_series = train_series - series_mean
            
            # 尝试拟合模型
            if use_seasonal:
                model = SARIMAX(
                    normalized_series,
                    order=order,
                    seasonal_order=use_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='c'  # 添加常数项
                )
            else:
                model = ARIMA(
                    normalized_series,
                    order=order,
                    trend='c'  # 添加常数项
                )
            
            # 使用更宽松的拟合参数
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                fitted_model = model.fit(
                    disp=False,
                    maxiter=100,  # 限制迭代次数
                    method='lbfgs'  # 使用不同的优化方法
                )
            
            # 检查模型收敛状态
            if hasattr(fitted_model, 'mle_retvals') and fitted_model.mle_retvals is not None:
                if not fitted_model.mle_retvals.get('converged', True):
                    print(f"ARIMA模型未完全收敛，但使用当前结果")
            
            # 预测并反标准化
            forecast = fitted_model.forecast(1)
            forecast_value = float(forecast.iloc[0])
            
            # 反标准化
            if series_std > 0:
                forecast_value = forecast_value * series_std + series_mean
            
            return max(forecast_value, 0.1)
            
        except Exception as e:
            print(f"ARIMA预测错误: {e}")
            # 回退到简单方法
            return float(np.mean(train_series[-96:]))
    
    def simple_arima_forecast(self, train_series):
        """极简ARIMA预测 - 最高稳定性"""
        print('simple_arima_forecast')
        try:
            # 使用最简单的可行参数
            model = ARIMA(
                train_series,
                order=(1, 0, 0),  # 简单AR(1)模型
                trend='c'
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fitted_model = model.fit(disp=False)
            
            forecast = fitted_model.forecast(1)
            return float(forecast.iloc[0])
            
        except:
            return float(np.mean(train_series[-48:]))
    
    def xgboost_forecast(self, train_data, test_features, demographic):
        """XGBoost 预测 (论文中提到的最佳方法)"""
        try:
            # 准备训练数据
            demo_train = train_data[train_data['demographic'] == demographic].copy()
            demo_train = demo_train.dropna()
            
            if len(demo_train) < 50:  # 数据太少时使用简单方法
                return self.historical_average(train_data, None, demographic, test_features['quarter_hour'].iloc[0])
            
            feature_cols = [col for col in demo_train.columns if col not in 
                          ['timestamp', 'demographic', 'rating', 'quarter_hour'] and 
                          not demo_train[col].isnull().all()]
            
            # 确保特征存在
            available_features = [col for col in feature_cols if col in demo_train.columns and col in test_features.columns]
            
            if not available_features:
                return self.historical_average(train_data, None, demographic, test_features['quarter_hour'].iloc[0])
            
            X_train = demo_train[available_features]
            y_train = demo_train['rating']
            
            # 处理缺失值
            X_train = X_train.fillna(X_train.mean())
            
            # 训练XGBoost模型 - 使用更稳定的参数
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,  # 降低深度以避免过拟合
                learning_rate=0.05,  # 降低学习率
                random_state=42,
                subsample=0.8,  # 添加正则化
                colsample_bytree=0.8
            )
            
            model.fit(X_train, y_train)
            
            # 预测
            X_test = test_features[available_features].fillna(X_train.mean())
            prediction = model.predict(X_test)[0]
            return max(prediction, 0.1)
            
        except Exception as e:
            print(f"XGBoost预测错误 {demographic}: {e}")
            return self.historical_average(train_data, None, demographic, test_features['quarter_hour'].iloc[0])
    
    def evaluate_models(self, df, test_size=0.2):
        """评估不同预测方法"""
        print("开始模型评估...")
        
        # 按时间划分训练测试集
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_data = df_sorted.iloc[:split_idx].copy()
        test_data = df_sorted.iloc[split_idx:].copy()
        
        # 创建特征
        train_data = self.create_features(train_data)
        test_data = self.create_features(test_data)
        
        models_performance = {}
        
        # 选择部分人口分组进行评估以节省时间
        demo_samples = df['demographic'].unique()[:1]  # 评估前6个分组
        demo_samples = demo_samples[:5]# numpy
        
        for demo in demo_samples:
            print(f"评估人口分组: {demo}")
            demo_test = test_data[test_data['demographic'] == demo]
            
            if len(demo_test) == 0:
                continue
                
            predictions = {}
            actuals = []
            
            # 准备时间序列数据
            ts_data = self.prepare_time_series_data(train_data, demo)
            
            for idx, row in demo_test.iterrows():
                actual = row['rating']
                actuals.append(actual)
                
                # 不同方法的预测
                predictions.setdefault('Historical_Avg', []).append(
                    self.historical_average(train_data, row['timestamp'], demo, row['quarter_hour'])
                )
                
                # Holt-Winters (使用自适应版本)
                print('Holt-Winterss')
                if ts_data is not None and len(ts_data) > 48:  # 至少半天数据
                    try:
                        # 根据数据长度选择季节性周期
                        if len(ts_data) >= 672 * 2:  # 至少2周数据
                            hw_pred = self.holt_winters_forecast(ts_data, 1)
                        else:
                            hw_pred = self.simplified_holt_winters(ts_data, 1, seasonal_periods=96)
                        predictions.setdefault('Holt_Winters', []).append(hw_pred)
                    except Exception as e:
                        print(f"Holt-Winters预测错误 {demo}: {e}")
                        predictions.setdefault('Holt_Winters', []).append(
                            self.historical_average(train_data, row['timestamp'], demo, row['quarter_hour'])
                        )
                
                # ARIMA (使用稳健版本)
                print('ARIMA')
                if ts_data is not None and len(ts_data) > 30:
                    try:
                        arima_pred = self.simple_arima_forecast(ts_data)
                        predictions.setdefault('ARIMA', []).append(arima_pred)
                    except Exception as e:
                        print(f"ARIMA预测错误 {demo}: {e}")
                        # 尝试极简版本
                        try:
                            simple_pred = self.simple_arima_forecast(ts_data)
                            predictions.setdefault('ARIMA', []).append(simple_pred)
                        except:
                            predictions.setdefault('ARIMA', []).append(
                                self.historical_average(train_data, row['timestamp'], demo, row['quarter_hour'])
                            )
                
                # XGBoost
                print('XGBoost')
                predictions.setdefault('XGBoost', []).append(
                    self.xgboost_forecast(train_data, pd.DataFrame([row]), demo)
                )
            
            # 计算MAPE
            demo_performance = {}
            for model_name, preds in predictions.items():
                if len(preds) == len(actuals) and len(preds) > 0:
                    # 过滤无效预测
                    valid_actuals = []
                    valid_preds = []
                    for a, p in zip(actuals, preds):
                        if p is None:p=0.0
                        if a > 0 and float(p) > 0 and not np.isnan(p):
                            valid_actuals.append(a)
                            valid_preds.append(p)
                    
                    if len(valid_actuals) > 0:
                        mape = mean_absolute_percentage_error(valid_actuals, valid_preds)
                        demo_performance[model_name] = mape
            
            models_performance[demo] = demo_performance
            break
        
        return models_performance
    
    def find_best_model_per_series(self, performance_results):
        """为每个时间序列选择最佳模型"""
        best_models = {}
        
        for demo, performance in performance_results.items():
            if performance:
                # 找到MAPE最小的模型
                valid_models = {k: v for k, v in performance.items() if not np.isnan(v)}
                if valid_models:
                    best_model = min(valid_models.items(), key=lambda x: x[1])
                    best_models[demo] = {
                        'best_model': best_model[0],
                        'mape': best_model[1]
                    }
        
        return best_models
    
    def forecast_ratings(self, historical_data, forecast_horizon=96):
        """
        执行最终预测 - 使用每个序列的最佳模型
        对应论文中提到的策略：为每个组合选择历史MAPE最优的模型
        """
        print("开始收视率预测...")
        
        # 准备数据
        full_data = self.create_features(historical_data)
        
        forecasts = {}
        
        # 选择部分分组进行预测演示
        demo_samples = historical_data['demographic'].unique()[:4]
        
        for demo in demo_samples:
            demo_data = full_data[full_data['demographic'] == demo].copy()
            
            if demo not in self.best_models:
                # 如果没有预计算的最佳模型，使用XGBoost作为默认
                model_to_use = 'XGBoost'
            else:
                model_to_use = self.best_models[demo]['best_model']
            
            print(f"为 {demo} 使用 {model_to_use} 模型进行预测")
            
            # 准备时间序列数据
            ts_data = self.prepare_time_series_data(demo_data, demo)
            
            # 根据选择的模型进行预测
            if model_to_use == 'Historical_Avg':
                # 历史平均预测
                demo_forecasts = []
                last_timestamp = demo_data['timestamp'].iloc[-1]
                
                for i in range(forecast_horizon):
                    future_time = last_timestamp + timedelta(minutes=15*(i+1))
                    quarter_hour = (future_time.hour * 4 + future_time.minute // 15) % 96
                    forecast_val = self.historical_average(demo_data, future_time, demo, quarter_hour)
                    demo_forecasts.append(forecast_val)
                
                forecasts[demo] = demo_forecasts
                
            elif model_to_use == 'XGBoost':
                # XGBoost预测 - 需要创建未来特征
                demo_forecasts = []
                last_row = demo_data.iloc[-1:].copy()
                
                for i in range(forecast_horizon):
                    # 更新时间特征
                    future_time = last_row['timestamp'].iloc[0] + timedelta(minutes=15*(i+1))
                    future_features = last_row.copy()
                    future_features['timestamp'] = future_time
                    future_features['hour'] = future_time.hour
                    future_features['day_of_week'] = future_time.dayofweek
                    future_features['is_weekend'] = 1 if future_time.dayofweek >= 5 else 0
                    future_features['quarter_hour'] = (future_time.hour * 4 + future_time.minute // 15) % 96
                    
                    # 重新计算时间周期特征
                    future_features['hour_sin'] = np.sin(2 * np.pi * future_features['hour'] / 24)
                    future_features['hour_cos'] = np.cos(2 * np.pi * future_features['hour'] / 24)
                    future_features['day_sin'] = np.sin(2 * np.pi * future_features['day_of_week'] / 7)
                    future_features['day_cos'] = np.cos(2 * np.pi * future_features['day_of_week'] / 7)
                    
                    # 填充其他特征
                    for col in future_features.columns:
                        if future_features[col].isnull().any():
                            future_features[col] = future_features[col].fillna(demo_data[col].mean())
                    
                    forecast_val = self.xgboost_forecast(demo_data, future_features, demo)
                    demo_forecasts.append(forecast_val)
                
                forecasts[demo] = demo_forecasts
                
            else:
                # 对于传统时间序列模型
                demo_forecasts = []
                
                if ts_data is not None and len(ts_data) > 0:
                    for i in range(forecast_horizon):
                        if model_to_use == 'Holt_Winters':
                            # 使用自适应Holt-Winters
                            if len(ts_data) >= 672 * 2:
                                forecast_val = self.holt_winters_forecast(ts_data, 1)
                            else:
                                forecast_val = self.simplified_holt_winters(ts_data, 1, 96)
                        elif model_to_use == 'ARIMA':
                            # 使用稳健ARIMA
                            forecast_val = self.robust_arima_forecast(ts_data)
                        else:
                            # 默认使用近期平均
                            forecast_val = np.mean(ts_data[-24:])
                        
                        demo_forecasts.append(forecast_val)
                        
                        # 模拟更新序列（在实际应用中应该重新拟合模型）
                        if isinstance(ts_data, pd.Series):
                            new_index = ts_data.index[-1] + timedelta(minutes=15)
                            ts_data = pd.concat([ts_data, pd.Series([forecast_val], index=[new_index])])
                else:
                    # 回退到历史平均
                    recent_avg = demo_data['rating'].mean()
                    demo_forecasts = [recent_avg] * forecast_horizon
                
                forecasts[demo] = demo_forecasts
        
        return forecasts

def main():
    """主函数 - 演示完整的预测流程"""
    print("=== 电视收视率预测系统 (基于论文实现) ===")
    
    # 初始化预测器
    forecaster = TelevisionRatingsForecaster(demographic_groups=16)
    
    # 1. 生成示例数据
    print("1. 生成模拟收视率数据...")
    ratings_data = forecaster.generate_sample_data(periods=672)  # 适中的数据量，一周的数据量96*7
    print(f"数据形状: {ratings_data.shape}")
    print(f"人口分组数量: {ratings_data['demographic'].nunique()}")
    
    # 2. 模型评估
    print("\n2. 评估不同预测方法...")
    performance_results = forecaster.evaluate_models(ratings_data, test_size=0.15)
    
    # 3. 为每个序列选择最佳模型
    print("\n3. 选择每个序列的最佳模型...")
    best_models = forecaster.find_best_model_per_series(performance_results)
    forecaster.best_models = best_models
    
    # 打印模型选择结果
    model_distribution = {}
    for demo, info in best_models.items():
        model_name = info['best_model']
        model_distribution[model_name] = model_distribution.get(model_name, 0) + 1
        print(f"  {demo}: {model_name} (MAPE: {info['mape']:.3f})")
    
    print("\n最佳模型分布:")
    for model, count in model_distribution.items():
        print(f"  {model}: {count} 个序列")
    
    # 计算平均MAPE
    if best_models:
        avg_mape = np.mean([info['mape'] for info in best_models.values()])
        print(f"\n平均MAPE: {avg_mape:.3f}")
        baseline_improvement = (0.44 - avg_mape) / 0.44 * 100
        print(f"相比历史平均基准(44%)提升: {baseline_improvement:.1f}%")
    
    # 4. 执行预测
    print("\n4. 执行未来48个时段(12小时)的预测...")
    forecasts = forecaster.forecast_ratings(ratings_data, forecast_horizon=48)
    
    # 5. 可视化结果
    print("\n5. 生成预测可视化...")
    sample_demos = list(forecasts.keys())[:min(4, len(forecasts))]
    print(forecasts)
    plt.figure(figsize=(15, 10))
       
    
    for i, demo in enumerate(sample_demos, 1):
        plt.subplot(2, 2, i)
        
        # 历史数据
        demo_historical = ratings_data[ratings_data['demographic'] == demo]
        demo_historical = demo_historical.sort_values('timestamp')
        historical_ratings = demo_historical['rating'].values[-96:]
        
        plt.plot(range(len(historical_ratings)), historical_ratings, 
                label='历史数据', color='blue', alpha=0.7, linewidth=2)
        
        # 预测数据
        forecast_vals = forecasts[demo]
        forecast_start = len(historical_ratings)
        forecast_end = forecast_start + len(forecast_vals)
        
        plt.plot(range(forecast_start, forecast_end), 
                forecast_vals, label='预测数据', color='red', linestyle='--', linewidth=2)
        
        # 添加分界线
        plt.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.7)
        
        model_info = forecaster.best_models.get(demo, {'best_model': 'XGBoost', 'mape': 0.0})
        plt.title(f'{demo}\n模型: {model_info["best_model"]} (MAPE: {model_info["mape"]:.3f})')
        plt.xlabel('时段 (15分钟)')
        plt.ylabel('收视率')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ratings_forecast_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # 6. 输出预测统计
    print("\n6. 预测结果统计:")
    for demo in sample_demos:
        forecast_vals = forecasts[demo]
        forecast_mean = np.mean(forecast_vals)
        forecast_std = np.std(forecast_vals)
        cv = forecast_std / forecast_mean if forecast_mean > 0 else 0
        print(f"{demo}: 均值={forecast_mean:.2f}, 标准差={forecast_std:.2f}, 变异系数={cv:.2f}")

if __name__ == "__main__":
    main()