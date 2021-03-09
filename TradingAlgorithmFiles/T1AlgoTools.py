


import alpaca_trade_api as tradeapi                                         # library used for Alpaca API
import pandas as pd                                                         # tools for DataFrames
from datetime import datetime, timedelta                                    # for working with dates
import os                                                                   # for collecting environment variables
from dotenv import load_dotenv                                              # for loading environment variables
import numpy as np                                                          # numeric array tools
import seaborn as sns                                                       # for plotting style
sns.set_style("darkgrid")                                                   # custom-color grid
#import matplotlib.pyplot as plt                                             # for plotting
#import hvplot.pandas                                                       # for constructing and analyzing neural networks



load_dotenv()                                                               # Load .env environment variables


alpaca_api_key = os.getenv("ALPACA_API_KEY")                                # Set Alpaca API key
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")                          # Set Alpaca secret key


alpaca = tradeapi.REST(alpaca_api_key,alpaca_secret_key,api_version="v2")   # Create the Alpaca API object


#from sqlalchemy import create_engine                                       # used to pull data from sql server

### Pulling the Dictionary From SQL
#db_url = "postgresql://postgres:postgres@localhost:5432/agent_db"
#mk = pd.read_sql(query, engine)




#"""-----------------           ALGO TOOLS !!!!           ---------------"""
#"""  Listed below are the functions that we will use to make the code more condense and clean""

##############################                                       #######################################
##############################               AROON FUNCTION          #######################################
#############################                                         #######################################

def aroon(df,period=14,option=1):
    """ex:A_up=aroon(candle_df,20,1)
          A_down=aroon(candle_df,20,-1)
    """
    # Make vector of zeros
    A =df['close'].copy()*0.0
    if(option==1):
        for i in range(period,len(df)):
            HH_index=np.where(df['high'][i-period:i].values==np.max(df['high'][i-period:i].values))   # gets index of highest high
            A[i]=100.0*float(1+HH_index[0][0])/float(period)              # value of the aroon up indicator; note the use of the zero index, in case max occurs at two positions
    elif (option==-1):
        for i in range(period,len(df)):
            LL_index=np.where(df['low'][i-period:i].values==np.min(df['low'][i-period:i].values))     # gets index of lowest low
            A[i]=100.0*float(1+LL_index[0][0])/float(period)              # value of the aroon up indicator; note the use of the zero index, in case the max occurs at two positions
    else:
        print("Invalid value for option parameter")   
    A[0:period]=np.NAN                                              # fills previous values with NAs, since we cannot have values for indices prior to period value
    return A                                                        # returns vector of aroon indicator values; note the conversion of certain values to floats













#############################                                    #####################################
##############################            ATR FUNCTION          ####################################### 
###########################.                                     ######################################
def ATR(df,period=14):
    """Computes the ATR indicator values, given prices of currency pair
    ex: ATR(df).hvplot(color='red',width=2000,height=500, title="ATR Indicator")"""
    ATR =df['close'].copy()*0.0                                        # making an zero DataFrame
    ATR_final =df['close'].copy()*0.0 
    # Comoputing the ATR values
    for i in range(len(df)):
        ATR[i]=np.max([df['high'][i]-df['low'][i],df['high'][i]-df['close'][i],df['low'][i]-df['close'][i]])
    ATR_final=ATR.rolling(period).mean()
    #ATR_final[0:period]=np.NAN                                        # filling entries before period index with NAs
    return ATR_final                                                   # returning the vector of CMO values










#############################                                                    ######################################
##############################            STOCHASTIC OSCILLATOR FUNCTION         #######################################
#############################.                                                   #######################################
def stoch_osc(df,fast_period=14,slow_period=3):
    """Function computes the value for the Stochastic Oscillator Function
    df: represents DataFrame of candlestick data
    fast_period: period for computing fast stochastic curve
    slow_period: period for computing slow stochastic curve"""
    fast_stochastic=df['close'].copy()*0.0
    slow_stochastic=df['close'].copy()*0.0
    for i in range(fast_period,len(df)):
        C=df['close'][i]                                             # current closing price
        L=np.min(df['low'][i-fast_period:i])                         # period low
        H=np.max(df['high'][i-fast_period:i])                        # period high
        fast_stochastic[i]=100.0*(C-L)/(H-L)                         # computing fast stochastic values
    fast_stochastic[0:fast_period]=np.NAN                            # filling entries before period index with NAs
    slow_stochastic.rolling(slow_period).mean()                      # get rolling average of fast stochastic values
    return fast_stochastic, slow_stochastic                          # returning the DataFrames
    
    
    
    

    
    

    
####################################     DATA CLEAN FUNCTION       #########################################  
def data_clean(df):
    """Cleans a data set and removes all None/NA entries"""
    if( sum(df.isnull().sum())>0 ):                                   # checks if there are any missing values
        df.fillna(method='ffill')                                     # fills entries using forward fill
    return(df)                                                        # returns the DataFrame
        
    

    
    

    
    
    
#Note: since buy and sell check are looping through stocks, the strategies only need to act on the single stocks 
# and not the entire portfolio
##############################                                            #####################################
##############################            STRATEGY 1                     ####################################### 
##############################                                           ######################################  
def strategy1(portfolio,ticker,check="buy"):
    """Checks if certain conditions are met to execute a buy or sell order,
    based on indicators used in strategy 1. This strategy uses the crossing of
    a SMA and EMA curve.
    stock_df: DataFrame containing basis price data on specific stock"""
    ### Values of indicators will depend on the value of "check" parameter
    result=False                                                                      # initializing the value of result variable
    if (check=="buy"):
        if((portfolio[ticker]["SMA-30"][-1]<portfolio[ticker]["EMA-10"][-1]) and 
        (portfolio[ticker]["SMA-30"][-2]>portfolio[ticker]["EMA-10"][-2])) :
            result=True                                                               # confirms that the signal has occurred
    elif (check=="sell"):
        if((portfolio[ticker]["SMA-30"][-1]>portfolio[ticker]["EMA-10"][-1]) and 
        (portfolio[ticker]["SMA-30"][-2]<portfolio[ticker]["EMA-10"][-2])) :
            result=True                                                               # confirms that the signal has occurred
        #somthing
    else:
        print("Not a valid value for check")
    return result                                                                     # returns confirmation that signal has occurred
        
         

        
    
##############################                                            #####################################
##############################            CLOSING TEST1                    ####################################### 
##############################                                           ###################################### 
def closing_test1(portfolio, ticker,side="buy"):
    """Checks to see if trade needs to be closed based on specified criteria"""
    result=False                                                                      # initializing the value of result variable
    ### Values of indicators will depend on the value of "side" parameter
    if (side=="buy"):
        if((portfolio[ticker]["SMA-30"][-1]>portfolio[ticker]["EMA-10"][-1]) and 
        (portfolio[ticker]["SMA-30"][-2]<portfolio[ticker]["EMA-10"][-2])) :
            result=True                                                               # confirms that the signal has occurred
    elif (side=="sell"):
        if((portfolio[ticker]["SMA-30"][-1]<portfolio[ticker]["EMA-10"][-1]) and 
        (portfolio[ticker]["SMA-30"][-2]>portfolio[ticker]["EMA-10"][-2])) :
            result=True                                                               # confirms that the signal has occurred
    else:
        print("Not a valid value for check")
    return result                                                                     # returns confirmation that signal has occurred
        
         
    
        
        
        
 #closing_test1(self.data,ticker,check=order_dict[ticker])       
        
        
        
        
        
        
        
    
    
###############################################                                                                           ###################################################################################
###############################################                 DEFINING THE CLASS USED IN PORTFOLIO ANALYSIS.            ###################################################################################
###############################################                                                                           ###################################################################################
class Portfolio:
    """ This class is for pulling financial records for specfic companies for a specified time range
        
     Attributes
        tickers=contains ticker symbols of specified companies
        fin_stat_items=items extracted from financial statements
        balance_sheet_items=items extracted from balance sheet statements
        fin_ratios_items=items extracted from ratios statements
        cash_flow_items=items extracted from cash flow statements
    """
###############    Initializing the Function
###############
    def __init__(self, tickers=["AAPL"],timeframe='15Min',start=(datetime.now()-timedelta(days=1095)).strftime("%Y-%m-%d"),end=datetime.now().strftime("%Y-%m-%d")):  
        """initializes the class with tickers for portfolio, timeframe, starting and end dates of scope
        timeframe"""
###############     Checking for Errors
        if not isinstance(tickers,list):
            raise TypeError("ticker must be a list")
#Setting class attributes
        self.tickers=tickers
        self.data=pd.DataFrame()                                                                                         # initializes data as empty DataFrame
        self.timeframe=timeframe                                                                                         # storing the timeframe value
        self.start=pd.Timestamp(start, tz="America/New_York").isoformat()                                                # storing the start time
        self.end=pd.Timestamp(end, tz="America/New_York").isoformat()                                                    # storing the end time
        
        
        
        
        
              
        
############################.                                                  ##############################
#############################          GET DATA FUNCTION  (CLEAR)              ##############################
#############################.                                                 ##############################

    def get_data_custom(self):
        """Gets data for tickers given by user from Alpaca in specified date range
        and specified timeframe. Result will be a single DataFrame stored in self.data attribute"""
        df_portfolio = alpaca.get_barset(
        self.tickers,
        self.timeframe,
        start = self.start,
        end = self.end
        ).df
        self.data=data_clean(df_portfolio)                                                     # stores cleaned DataFrame of stock information
    
    
    def get_data_recent(self):
        """Gets data contained in the most recent 100 candles oftickers given by user from Alpaca in specified date range
        and specified timeframe. Result will be a single DataFrame stored in self.data attribute"""
        df_portfolio = alpaca.get_barset(
        self.tickers,
        self.timeframe,
        limit=100,
        ).df
        self.data=data_clean(df_portfolio)                                                    # stores cleaned DataFrame of stock information
  


    
############################.                                                                   ##############################
#############################          GET INDICATORS FUNCTION (WORK IN PROGRESS)                 ###############################
#############################.                                                                    ##############################    
    

            
            
            
            
            
        

##############################                                          ######################################
##############################             ROC FUNCTION (CLEAR)         #######################################
##############################                                          #######################################
    def ROC(self,p=14):
        """Takes in a dataframe of forex prices and computes the values of the ROC indictor for the specified period
        df: DataFrame of candlestick data
        p: period used in computation of ROC indicator"""
        #p-value of the computational period
        for ticker in self.tickers:
            self.data[(ticker,"ROC"+"-"+str(p))]=100*self.data[(ticker,'close')].diff(p)/self.data[(ticker,'close')][p:]# constructing the ROC DataFrame
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                     # updating DataFrame after reindexing MultiIndex
    
   








 #############################                                                           ######################################
##############################            STOCHASTIC OSCILLATOR FUNCTION (CLEAR)         #######################################
#############################.                                                           #######################################
    def stoch_osc(self,fast_period=14,slow_period=3):
        """Function computes the value for the Stochastic Oscillator Function
        df: represents DataFrame of candlestick data
        fast_period: period for computing fast stochastic curve
        slow_period: period for computing slow stochastic curve"""
        for ticker in self.tickers:
            fast_stochastic=self.data[(ticker,'close')].copy()*0.0                                                      # initializing vectors for stochastic indicator
            #slow_stochastic=self.data[(ticker,'close')].copy()*0.0
            for i in range(fast_period,len(self.data)):
                C=self.data[(ticker,'close')][i]                                                                        # current closing price
                L=np.min(self.data[(ticker,'low')][i-fast_period:i])                                                    # period low
                H=np.max(self.data[(ticker,'high')][i-fast_period:i])                                                   # period high
                fast_stochastic[i]=100.0*(C-L)/(H-L)                                                                    # computing fast stochastic values
            fast_stochastic[0:fast_period]=np.NAN                                                                       # filling entries before period index with NAs
            slow_stochastic=fast_stochastic.rolling(slow_period).mean()                                                 # get rolling average of fast stochastic values
            self.data[(ticker,"stoch_fast")]=fast_stochastic                                                            # storing the stochastic fast data
            self.data[(ticker,"stoch_slow")]=slow_stochastic                                                            # storing the stochastic slow data
    #return fast_stochastic, slow_stochastic                                                                            # returning the DataFrames
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                     # updating DataFrame after reindexing MultiIndex
    
    
    
    

    
    
    
#############################                                            ######################################
##############################            RSI FUNCTION (CLEAR)           #######################################
#############################.                                           #######################################
    def RSI(self, period=14):
        """Function computes the values for the RSI indicator
        df: DataFrame of candlestick data
        period: period used in computation of rsi indicator (14 default)"""
        for ticker in self.tickers:
            RSI=self.data[("AAPL",'close')].copy()*0.0                                                                  # allocating space for the RSI indicator         
            diffs=self.data[("AAPL",'close')].diff()                                                                    # computing differences of closing price over 1 period
            pos_diff=pd.DataFrame(np.where(diffs>0.0,diffs,0.0))                                                        # finds where price difference is positive
            neg_diff=pd.DataFrame(np.where(diffs<0.0,-diffs,0.0))                                                       # finds where the price difference is negative
            PA=pos_diff.rolling(period).mean()                                                                          # average positive change
            NA=neg_diff.rolling(period).mean()                                                                          # average negative change
            for i in range(period,len(RSI)):                                                                            # loop for computing the RSI values
                RSI.iloc[i]=100.0-100/(1+(PA.iloc[i].values/NA.iloc[i].values))                                         # computing the RSI values
            self.data[(ticker,"RSI"+"-"+str(period))]=RSI                                                               # storing the RSI value
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                     # updating DataFrame after reindexing MultiIndex



        
        
        
    
    
    
    
#############################                                           ######################################
##############################            CMO FUNCTION (CLEAR)          #######################################
#############################.                                          #######################################
    def CMO(self,period=14):
        """Function computes the values for the CMO"""
        for ticker in self.tickers:
            CMO =self.data[(ticker,'close')].copy()*0.0                                                                 # making an zero DataFrame
            for i in range(period,len(self.data)): 
                close_prices=self.data[(ticker,'close')][i-period:i]                                                    # getting necessary values of closing price
                A=close_prices.diff()                                                                                   # getting differences in closing prices over 1 period
                SU=A[A.values>0].values.sum()                                                                           # getting all positive  sums
                SD=A[A.values<0].values.sum()*-1                                                                        # getting all negative sums
                CMO[i]=100*(SU-SD)/(SU+SD)                                                                              # computation of CMO value
            CMO[0:period]=np.NAN                                                                                        # filling entries before period index with NAs
            self.data[(ticker,"CMO"+"-"+str(period))]=CMO                                                               # storing the CMO results
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                     # updating DataFrame after reindexing MultiIndex


        
        
    
    
        
        
##############################                                                        #######################################
##############################            CHANDE KROLL STOP FUNCTION (CLEAR)          #######################################
##############################                                                       ######################################
    def CKS(self,p=10,q=9,x=1,option=1):
        """Description: Computes values for the Chande Kroll Stop function, for both short and long positions.
        These will ideally be used constructing stop loss levels for trades
        ex: cks(candle_df,10,9,1,1).head(30)"""
        for ticker in self.tickers:
            CKS=self.data[(ticker,'close')].copy()*0.0                                                                 # making an zero DataFrame
            atr_df=ATR(self.data[ticker],p)                                                                            # computes ATR values
            if(option==1):                                                                                             # stop loss for long positions
                for i in range(p+q,len(self.data)):                                                                           # loop for iterating over values of currecny pair
                # compute preliminary lows
                    prelim_low_vals=[np.min(self.data[(ticker,'low')][i-j-p:i-j]+x*atr_df[i-j]) for j in range(q)]  
                
                # pick smallest preliminary low
                    CKS[i]=min(prelim_low_vals)                 
            elif (option==-1):                                                                                         # stop loss for short positions
                for i in range(p+q,len(self.data)):                                                                    # loop for iterating over values of currecny pair
                # compute preliminary highs
                    prelim_high_vals=[np.max(self.data[(ticker,'high')][i-j-p:i-j]-x*atr_df[i-j]) for j in range(q)]  
                
                # pick largest preliminary high
                    CKS[i]=max(prelim_high_vals)   
            else:
                print("Invalid value for option parameter")                                                            # when an invalid parameter is entered
        
    # Filling in rest of vector
            CKS[0:p+q]=np.NAN                                                                                          # filling entries before period index with NAs
            self.data[(ticker,"CKS")]=CKS                                                                              # storing the CKS values
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                    # updating DataFrame after reindexing MultiIndex


        

        
        
    
    
    
        
##############################                                          #####################################
##############################            MA FUNCTION (CLEAR)           ####################################### 
#############################                                           ######################################
    def MA(self, period=14):
        """Computes the moving average(MA) indicator values, given the prices of currency pair
        df: DataFrame of candlestick information
        period: period used in computing moving average"""
        for ticker in self.tickers:
            self.data[(ticker,"SMA"+"-"+str(period))]=self.data[(ticker,'close')].rolling(period).mean()               # computes MA indicator and storing it
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                    # updating DataFrame after reindexing MultiIndex



        
        

##############################                                            #####################################
##############################            EMA FUNCTION  (CLEAR)          ####################################### 
##############################                                           ######################################
    def EMA(self, period=14):
        """Computes the moving average(MA) indicator values, given the prices of currency pair
        df: DataFrame of candlestick information
        period: period used in computing moving average"""
        for ticker in self.tickers:
            self.data[(ticker,"EMA"+"-"+str(period))]=self.data[(ticker,'close')].ewm(period).mean()                   # computes MA indicator
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                    # updating DataFrame after reindexing MultiIndex



        
        
        
        
##############################                                               #####################################
##############################            BOLLINGER BAND FUNCTION (CLEAR)    ####################################### 
##############################.                                              ######################################   
    def bb(self,bollinger_window=14,dev=2):
        """Function computes the upper, middle and lower bollinger bands to use
        as indicators.
        bollinger_window: period used to compute the bollinger bands
        dev: number of standard deviations away from middle band"""
        for ticker in self.tickers:
            self.data[(ticker,'bollinger_mid_band')] = self.data[(ticker,'close')].rolling(window=bollinger_window).mean() # computing the middle band of bollinger bands
            std_vec = self.data[(ticker,'close')].rolling(window=bollinger_window).std()                                   # computing the standard deviation

            self.data[(ticker,'bollinger_upper_band')]  = self.data[(ticker,'bollinger_mid_band')] + (std_vec * dev)       # calculate upper band
            self.data[(ticker,'bollinger_lower_band')]  = self.data[(ticker,'bollinger_mid_band')] - (std_vec * dev)       # calculate lower band
        
        self.data=self.data.reindex(columns=self.data.columns.sortlevel(0,sort_remaining=False)[0])                        # updating DataFrame after reindexing MultiIndex
        
              
    
    
    
###############################################                                                                           ###################################################################################
###############################################                 LISTING OF TRADING CLASSES                                ###################################################################################
###############################################                                                                           ###################################################################################            
   

    
        

        
        
        
        
##############################                                            #####################################
##############################            STRATEGY 2                     ####################################### 
##############################                                           ######################################  
    def strategy1(self,check="buy"):
        """Checks if certain conditions are met to execute a buy or sell order,
        based on indicators used in strategy 1"""
        ### Values of indicators will depend on the value of "check parameter"        
        
        
    
    
    
    
    
    
###############################################                                                                           ###################################################################################
###############################################                 EXECUTING TRADES                                          ###################################################################################
###############################################                                                                           ###################################################################################        
 

        
        
    
    
        
        
##############################                                      #####################################
##############################            BUY CHECK                 ####################################### 
##############################.                                     ######################################
    def buy_check(self):
        """Function checks for buy opportunities from stocks in portfolio.
        strategy must be provided which will be used to check for buy opportunities.
        The necessary indicators must be present in order for the strategy to be executed"""
        portfolio = alpaca.list_positions()                                                                               # Get a list of all of our positions.
        open_order_tickers=[position.symbol for position in portfolio]                                                 # Gets list of symbols from open orders    
        
        free_tickers=[tick for tick in self.tickers if tick not in open_order_tickers]                                 # list tickers that don't have open orders
        for ticker in free_tickers:                                                                            # loops through available tickers
            result=strategy1(self.data,ticker,check="buy")
            if(result==True):
            #provided that a buy opportunity exists
                print("Trade Made!")
                #alpaca.submit_order(
                #symbol=ticker,
                #qty=50,
                #side='buy',
                #type='trailing_stop',
                #trail_percent=1.0,  # stop price will be hwm*0.99
                #time_in_force='gtc',
                #)
            else:
                print("Trade not Made!")
                
        

        
    
##############################                                      #####################################
##############################            SELL CHECK                 ####################################### 
##############################.                                     ######################################
    def sell_check(self):
        """Function checks for buy opportunities from stocks in portfolio.
        strategy must be provided which will be used to check for buy opportunities.
        The necessary indicators must be present in order for the strategy to be executed"""
        portfolio = alpaca.list_positions()                                                                               # Get a list of all of our positions.
        open_order_tickers=[position.symbol for position in portfolio]                                                 # Gets list of symbols from open orders    
        
        free_tickers=[tick for tick in self.tickers if tick not in open_order_tickers]                                 # list tickers that don't have open orders
        for ticker in free_tickers:                                                                            # loops through available tickers
            result=strategy1(self.data,ticker,check="sell")
            if(result==True):
            #provided that a buy opportunity exists
                print("Trade Made!")
                #alpaca.submit_order(
                #symbol=ticker,
                #qty=50,
                #side='buy',
                #type='trailing_stop',
                #trail_percent=1.0,  # stop price will be hwm*0.99
                #time_in_force='gtc',
                #)
            else:
                print("Trade not Made!")
                
                
 ##############################                                      #####################################
##############################             CLOSE CHECK                 ####################################### 
##############################.                                     ######################################
    def close_check(self):
        """Function checks for buy opportunities from stocks in portfolio.
        strategy must be provided which will be used to check for buy opportunities.
        The necessary indicators must be present in order for the strategy to be executed"""
        portfolio = alpaca.list_positions()                                                                      # Get a list of all of our positions.
        open_order_tickers=[position.symbol for position in portfolio]                                           # Get list of symbols from open orders  
        open_order_sides=[position.side for position in portfolio]                                               # get side for each symbol from open order
        order_dict={open_order_tickers[i]:open_order_sides[i] for i in range(len(open_order_tickers))}           # creating dictionary
        
        #free_tickers=[tick for tick in self.tickers if tick not in open_order_tickers]                          # list tickers that don't have open orders
        for ticker in open_order_tickers:                                                                        # loops through available tickers
            result=closing_test1(self.data,ticker,side=order_dict[ticker])                                      # checks if position is ready to close
            if(result==True):
            #provided that a buy opportunity exists
                print("Trade Closed!")
                #alpaca.close_position(symbol=ticker)                                                             # closes trade corresponding to given ticker
            else:
                print("Trade not Closed !")
                


        
# For the close_check() function, we need to consider the side of the trade and the quantity of share for each open posiiton
        
    
#portfolio = alpaca.list_positions()   
#portfolio.side can either be long or short
# portfolio.qty represents the number of shares for the current position (short or long); make sure you do float(portfolio.qty)





