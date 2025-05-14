import pytest
import pandas as pd
from forecast import load_data , tune_prophet , make_future_dataframe , generate_forecast


'''

Confirms and makes sure that the data is loaded in correctly from the CSV
Makes sure that no featutres were droppped or renamed - things like "ds", "humidity", and "y"
Confirms that the "ds" column was converted to datetime formating properly

'''
def test_loading_data():
    weather_df = load_data("weather.csv" ) 
    assert not weather_df.empty 
    assert "ds" in weather_df.columns
    assert "humidity" in weather_df.columns 
    assert "y" in weather_df.columns 
    
    assert pd.api.types.is_datetime64_any_dtype(weather_df[ "ds" ]) 
    

'''

Confirms that Prophet trains model without error
Checks we have "predict" function so ready to forecast/predict
Makes sure Optuna doesn't crash when tuning hyperparameters
Ensures model ready for forecasting

'''
    
    
def test_model_training() :
    weather_df = load_data( "weather.csv" ) 
    trained_model = tune_prophet( weather_df )
    assert hasattr( trained_model, "predict" ) #model able to predict
    
'''

Checks if the model creates a future df with new dates
Checks if that df has correct structure
Checks if it uses our extra features like "humidity" , "temp_rolling3" and "humidity_lag1"
Ensures have the "ds" for future forecast dates 

'''

def test_future_dataframe_generation():
    weather_df = load_data( "weather.csv" ) 
    trained_model = tune_prophet( weather_df )
    future_df = make_future_dataframe( trained_model, weather_df , periods= 7 )
    
    assert not future_df.empty  #not blank
    assert "ds" in future_df.columns #future dates are there
#columns used as regressors
    assert "humidity" in future_df.columns 
    assert "humidity_lag1" in future_df.columns 
    assert "temp_rolling3" in future_df.columns
    assert len( future_df ) > len( weather_df ) #have more rows than original df
    
  
'''

Checks if model genrates the forecast with all expected outputs
Including prediction with : confidence intervals for the next 7 days

'''
  
    
def test_output_forecast():
    weather_df = load_data( "weather.csv" ) 
    trained_model = tune_prophet( weather_df ) 
    future_df = make_future_dataframe( trained_model , weather_df , periods= 7 )       
    forecast_df = generate_forecast( trained_model , future_df )  
    
    assert not forecast_df.empty 
    assert "yhat" in forecast_df.columns 
    assert "yhat_lower" in forecast_df.columns  #lower confidence interval
    assert "yhat_upper" in forecast_df.columns  #upper confidence interval
# Double checking that all the NB columns exist
    for required_column in [ "ds", "yhat" , "yhat_lower" , "yhat_upper" ]:
        assert required_column in forecast_df.columns
    assert not forecast_df.tail( 7 ).empty
    