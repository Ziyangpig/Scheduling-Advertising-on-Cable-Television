from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

class RatingForecastModels:
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def historical_average(self, y_train, y_test, window=96):
        
        train_df = pd.DataFrame({'y': y_train.values})
        train_df['time_slot'] = train_df.index % window
        
        slot_means = train_df.groupby('time_slot')['y'].mean()
        
        test_time_slots = [i % window for i in range(len(y_test))]
        y_pred = [slot_means[slot] for slot in test_time_slots]
        
        self.models['historical_avg'] = slot_means
        self.predictions['historical_avg'] = y_pred
        
        return y_pred
    
    def holt_winters_model(self, y_train, y_test, start_date, seasonal_periods=96*7):
        
        y_series = pd.Series(y_train.values)
        index = pd.date_range(start=start_date, periods=len(y_series), freq='15T')
        y_series.index = index
        
        try:
            model = ExponentialSmoothing(
                y_series,
                seasonal_periods=seasonal_periods,
                trend='mul',
                seasonal='add',
                initialization_method='heuristic'
            ).fit()
            
            y_pred = model.forecast(len(y_test))
            for key, value in y_pred.items(): 
                y_pred[key] = np.clip(value, 0, 100)
            
            self.models['holt_winters'] = model
            self.predictions['holt_winters'] = y_pred
            
            return y_pred
        except Exception as e:
            print(f"Holt-Winters模型训练失败: {e}")
            return np.full(len(y_test), y_train.mean())
    
    def arima_model(self, y_train, y_test, start_date, order=(2,0,2), seasonal_order=(1,0,1,96)):

        y_series = pd.Series(y_train.values)
        index = pd.date_range(start=start_date, periods=len(y_series), freq='15T')
        y_series.index = index
        
        try:
            model = ARIMA(
                y_series, 
                order=order
                #,seasonal_order=seasonal_order
            ).fit()
            
            y_pred = model.forecast(len(y_test))
            
            self.models['arima'] = model
            self.predictions['arima'] = y_pred
            
            return y_pred
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
            return np.full(len(y_test), y_train.mean())
    
    def xgboost_model(self, X_train, X_test, y_train, y_test):
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        
        self.models['xgboost'] = model
        self.predictions['xgboost'] = y_pred
        
        return y_pred
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        num = len(y_true)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        metrics = {
            'MAPE': mape,
            'MAE': mae,
            'RMSE': rmse
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def compare_models(self, y_test):
        comparison = {}
        
        for model_name, y_pred in self.predictions.items():
            if len(y_pred) == len(y_test):
                metrics = self.calculate_metrics(y_test, y_pred, model_name)
                comparison[model_name] = metrics
           
        comparison_df = pd.DataFrame(comparison).T
        comparison_df = comparison_df.round(4)
        
        return comparison_df
        
    def pred_to_csv(self, y_test, X_test):
         
        for model_name, y_pred in self.predictions.items():
            if model_name=='arima' or model_name == 'holt_winters':
               y_pred = y_pred.values
            df = pd.DataFrame({
                'date': X_test['date'],
                #'day_of_week': X_test['day_of_week'],
                'time': X_test['time'],
                'y_true':y_test,
                'y_pred':y_pred
            })
            if model_name=='arima' or model_name == 'holt_winters':print(f'y_prde:{type(y_pred)}')
            df.to_csv(f'{model_name}.csv',index = False)
            print(f"\n结果已保存到 '{model_name}.csv'")
            