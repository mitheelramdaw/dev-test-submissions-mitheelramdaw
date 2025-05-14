import optuna
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


'''
Loading CSV and preprocessing for Prophet
Check for misisng file, incorrect column names, and date formatting
Rename temperature_celsius to y for model
Smooth noise
Add lag and roling features'''


def load_data(file_path):
    if not os.path.exists(file_path): 
        raise FileNotFoundError( f"Unfortunately the file  {file_path} can't be found :( ")
    
    try: 
        df = pd.read_csv(file_path) 
    except Exception as read_error : 
        raise ValueError(f" 😬 Ooops, we ran into an error reading the CSV :{read_error} ")   
    
    if  "temperature_celsius" in df.columns :
        df.rename(columns= { "temperature_celsius": "y"}, inplace = True ) # prophet needs y column
        
        
    for column_name in [ "ds", "y", "humidity" ]:
        if column_name not in df.columns :
            raise ValueError(f" There is a missing column that is required : {column_name} " )
        
    try : 
        df[ "ds"]= pd.to_datetime(df[ "ds" ]) # prophets needs date column as ds format
    except Exception as date_error:
        raise ValueError( f"W ran into an error parsing the date 😬 : { date_error}  ")  
    
#Smoothiing temperature using ewma - Exponential weighted moving average
    df[ "y"]=df["y" ].ewm(span=5, adjust= False).mean() 
#Feature engineering : Adding lag and rolling features - Prophet gets memory of the past
    df["humidity_lag1"] =df["humidity"].shift(1) #Yesterday's humidity
    df["temp_rolling3"]=df["y"].rolling(3 ).mean() #3 day rolling avrg of smoothened temperature
    
#Filling misisng values as a precaution just incase there are any
    df.bfill( inplace = True) 
    df.ffill(inplace=True )
    return df

'''

Tuning Prophet parameter using Optuna
Using cross validation to get RMSE
Trying different paramters then returning the RMSE so it can pick the best one

'''


def objective(optuna_trial, df) :
    params={
        "changepoint_prior_scale" :optuna_trial.suggest_float( "changepoint_prior_scale" ,0.001,0.5,log =True), 
        "seasonality_prior_scale":optuna_trial.suggest_float("seasonality_prior_scale" , 0.01, 10, log=True),  
        "holidays_prior_scale": optuna_trial.suggest_float( "holidays_prior_scale" , 0.01, 10, log=True), 
        "seasonality_mode":optuna_trial.suggest_categorical( "seasonality_mode" , ["additive", "multiplicative"]), 
        "changepoint_range": optuna_trial.suggest_float("changepoint_range", 0.8, 0.95) 
    } 
    
# Try model with suggested parameters below:
    model = Prophet( **params )
    model.add_regressor( "humidity" )  
    model.add_regressor("humidity_lag1" ) 
    model.add_regressor( "temp_rolling3") 
    
    
    model.fit(df) 
    
#Cross validation to get RMSE
#Cross validation to see how well the combination is below:
    cv_results=cross_validation(model,initial="180 days" ,period="30 days" , horizon = "30 days",parallel= "threads" ) 
    metrics=performance_metrics(cv_results )
    return metrics[ "rmse" ].mean() 

'''

Find the best combination of paramater by Tuning with Optuna
Helps Propeht to not guess and perform better

'''




def tune_prophet(df):
    study=optuna.create_study(direction = "minimize") #Trying to minimise RMSE
    study.optimize(lambda optuna_trial : objective(optuna_trial , df ) , n_trials= 20 ) 
    print( "Best hyperparameters are : ", study.best_params )
    # print( "Best RMSE is : ", study.best_value ) 
    
#Re training Prohet based on the best parameters  
    model = Prophet(**study.best_params) 
    model.add_regressor("humidity" ) 
    model.add_regressor("humidity_lag1" ) 
    model.add_regressor("temp_rolling3" )   
    
    
    model.fit(df) 
    return model 

'''

Generate future df that Prophet will use to forecast
Re-attach regressors like:
                   humidity
                   temp_rolling3
                   etc
                   to the future dates
                     
'''



def make_future_dataframe( model, df , periods) :
    future_df = model.make_future_dataframe( periods= periods) 
    future_df= future_df.merge(df[[ "ds" , "humidity" , "humidity_lag1", "temp_rolling3"] ], on= "ds", how= "left" )
    
#If there is missing future regressor data or rolling info - use last known 
   
    for column_name in [ "humidity" , "humidity_lag1", "temp_rolling3" ] : 
        if column_name in future_df.columns:
            future_df[ column_name ]=future_df[column_name ].fillna( df[ column_name].iloc[ -1] )
            
    return future_df

'''

Using the trained model to generate the forecast 
Gets back full forecast df with the confidence intervals

'''

def generate_forecast( model, future_df ) :
    forecast_df = model.predict(future_df)
    required_columns = ["ds", "yhat" , "yhat_lower" , "yhat_upper" ] 
    for column in required_columns:
        if column not in forecast_df.columns:
            raise ValueError( f"There seems to be a missing column in the forecast : {column} ")
    return forecast_df

'''

Plot the full forecast - historic & future
Use the dark mode bg and blue for better aesthetics

'''

def plot_forecast(forecast_df , save_path= "Historical and Future Forecast.png" ):
    plt.style.use( "dark_background" )
    plt.figure(figsize=(10 , 5))
    plt.plot(forecast_df["ds"], forecast_df[ "yhat"], label= "Forecast", color= "deepskyblue" )
    plt.fill_between(forecast_df[ "ds"], forecast_df[ "yhat_lower"], forecast_df["yhat_upper"], alpha= 0.3, color= "dodgerblue")
    plt.xlabel("Date ", fontsize= 12, color = "lightgray" )
    plt.ylabel("Temperature in (°C)", fontsize = 12, color = "lightgray")
    plt.title("Forecast : Historic & Future 7 Days ", fontsize= 14, fontweight= "bold", color= "white")
    plt.legend(facecolor = "black", edgecolor="white" )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Chart has been saved to {save_path} ")
    plt.show()
    
'''

Plotting the next 7 days forecast
Close up of the forecast
Only the nexxt 7 days from the end of last date( 2023-12-31,12 )
Dark mode bg with blue for aesthetics

'''    

    
def next_7days_forecast_plot(forecast_df, save_path="Next 7 days Forecast.png"):
    plt.style.use("dark_background")
    plt.figure(figsize=(12, 6))
    plt.plot(
        forecast_df[ "ds"] ,
        forecast_df[ "yhat"], 
        label="7-Day Forecast", 
        marker="o" ,
        markersize= 8,
        markerfacecolor= "deepskyblue",
        markeredgecolor="deepskyblue" ,
        color= "deepskyblue",
        linewidth = 2
    )
    
    plt.fill_between(forecast_df["ds"] , forecast_df[ "yhat_lower"], forecast_df["yhat_upper"], alpha= 0.3, color= "dodgerblue")
    plt.xlabel("Date : ", fontsize=12, color="lightgray")
    plt.ylabel("Temperature (°C)", fontsize=12, color="lightgray")
    plt.title("Next 7 Days Temperature Forecast " , fontsize= 14, fontweight= "bold", color = "white")
    plt.legend(facecolor="black", edgecolor="white")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"The next 7 days forecast chart is saved to {save_path} ")
    plt.show()
    
'''

Main entry point for the script
Parses the CLI args and runs the entire flow of the forecast
Saves the Plots and CSV if specified
Plots:
    > Historical and Future Forecast
    > Next 7 Days Forecast


'''

    
def main():
    parser = argparse.ArgumentParser( description = " Forcasting weather with humidity and Prophet" )
    parser.add_argument( "--input" , type = str , required = True , help = " This is the Path to the CSV used for input... ") 
    parser.add_argument( "--periods" , type = int , default = 7 , help = " How many days you want to get the forecast for.... ") 
    parser.add_argument( "--output" , type = str , help = " This is if you want the option of saving the output CSV path... " ) 
    args = parser.parse_args( ) 
    
    try : 
        weather_df = load_data( args.input )
        trained_model = tune_prophet( weather_df)  
        future_df =make_future_dataframe( trained_model , weather_df , args.periods )
        forecast_df = generate_forecast( trained_model, future_df ) 
        
        forecast_only= forecast_df[ forecast_df["ds"] > weather_df["ds"].max() ].head( args.periods )
        print(forecast_only[ [ "ds", "yhat", "yhat_lower" , "yhat_upper" ] ] )
        
        
        if args.output :
            forecast_df.to_csv( args.output , index=False )
            print( f"The forecast has been saved to : {args.output} ") 
        
        
        plot_forecast( forecast_df ) 
        next_7days_forecast_plot( forecast_only ) 
        
    except Exception as forecast_error :
        print( f" 😬 Yikes we ran inot an unexpected error while trying to forecast : {forecast_error} " , file = sys.stderr )
        sys.exit( 1 ) 
        
        
        
        
        
if __name__ == "__main__" :
    main() 