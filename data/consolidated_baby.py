

# Imports -----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.pipeline import make_pipeline

#from datetime import datetime
#from datetime import timedelta
#Xtrapolate Functions
import random

#Bokeh Charts.py
from flask import Flask
from flask import Flask, render_template, request
from bokeh.embed import components
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, NumeralTickFormatter
# from xtrapolate_functions import scoring, auto_reg_lin
from bokeh.models import HoverTool


# Flask constructor ------------------------------------------------------------------------------------------------------
# Flask constructor ------------------------------------------------------------------------------------------------------
app = Flask(__name__)
counter = 0

selected_data_set = 'random'


start = 0


sv = 0


bn = 'Anna'

# Connect to Data --------------------------------------------------------------------------------------------------------
file_url = 'https://raw.githubusercontent.com/T-Rex-chess/OMSBACapstone/main/data/sales_data_sample.csv'
sales_data = pd.read_csv(file_url, encoding='ISO-8859-1') #Handles wider range of special char.
sales_data_df = pd.DataFrame(sales_data) # create a dataframe of the sales data
#print(sales_data.head())
#print("\n Sales Dataframe Head:")
#print(sales_data_df.head())

file_url2 = 'https://raw.githubusercontent.com/T-Rex-chess/OMSBACapstone/main/data/Namecount.csv'
babynames = pd.read_csv(file_url2, encoding='ISO-8859-1')


babynames = pd.DataFrame(babynames).set_index('Name')

babynames = babynames.T






# BaseManager Parent Class -----------------------------------------------------------------------------------------------
# BaseManager class: Parent class.
class BaseManager:
    def __init__(self, data):
        self.data = data





# DataLoader Class -------------------------------------------------------------------------------------------------------
# DataLoader class: Responsible for loading and preprocessing data, including handling NaN values and non-numeric columns.
class DataLoader(BaseManager):
    def load_data(self):
        print("Data loaded successfully.")
    
    def handle_missing_values(self):
        # Handle missing values for numeric columns by filling with the median
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        # Flag which datapoints had any missing values (for numeric colunms)
        for col in numeric_cols:
            self.data[f"{col}_missing"] = self.data[col].isna().astype(int)
        # Fill non-numeric missing values with "Unknown" or a similar placeholder
        self.data.fillna({'STATE': 'Unknown', 'TERRITORY': 'Unknown'}, inplace=True)
        print("Missing values handled successfully.")

    
    def preprocess_data(self, columns_to_drop, categorical_cols, dates):
        # Drop irrelevant columns
        #columns_to_drop = ['ORDERNUMBER', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
        if len(columns_to_drop) > 0:
            self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        # Encode categorical variables using one-hot encoding
        #categorical_cols = ['STATUS', 'PRODUCTLINE', 'DEALSIZE', 'COUNTRY']
        if len(categorical_cols) > 0:
            self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        # Drop non-numeric columns after encoding
        #convert dates to datetime
        if len(dates) > 0:
            self.data[dates] = pd.to_datetime(self.data[dates])
        #self.data = self.data.select_dtypes(include=[np.number, pd._libs.tslibs.timestamps.Timestamp])
        print("Non-numeric columns dropped and categorical variables encoded successfully.")
    

    def split_data(self, target_y_column, target_x_columns = None, test_size = 0.2):
        if target_x_columns == None:
            X = self.data.drop(columns=[target_y_column])
        else:
            cols_to_drop = []
            for i in list(self.data):
                
                if i in target_x_columns:
                    inx = 1
                else:
                    cols_to_drop.append(i)
                   
            X = self.data.drop(columns=cols_to_drop)
        y = self.data[target_y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    
    
    
    def get_baby(self, name = None, n = 50, test_value = -5, minsum = 99):
        global bn
        global start
        
        global sv

        if sv == 0:
            start = random.randint(1880, 2024 - n)
            if name == None:
                bnsum = 0
                while bnsum < minsum:
                    bn = random.choice(list(self.data.columns))
                    name_ser = self.data[bn].tolist()
                    df = name_ser[(start - 1880):(start+n-1880)]
                    bnsum = sum(df)
                
                
            else:
                bn = name
                
                name_ser = self.data[bn].tolist()

                df = name_ser[(start - 1880):(start+n-1880)]
        else:
            name_ser = self.data[bn].tolist()
            
            df = name_ser[(start - 1880):(start+n-1880)]
        
        

        

        x = list(range(start, (start+n)))
        
        #x = list(map(int, x))
    
        
        
        y_train = df[0:test_value]
        X_train = x[0:test_value]
        y_test = df[test_value:]
        X_test = x[test_value:]
        
        return X_train, X_test, y_train, y_test
            
    
    
    
    def ordered_split_data(self, target_y_column, target_x_columns = None, test_value = -5, cull = False, n = 35):
        global start
        global sv
        od = self.data

        if target_x_columns == None:
            od = od.sort_values(target_y_column)
            if cull == True:
                if sv == 0:
                    start = random.randint(0, len(od.index - (n+6)))
                    

            else: 
                n = n
            od = od.iloc[start:start+n+1]
            X = od.data.drop(columns=[target_y_column])
        else:
            c = target_x_columns[0]
            od = od.sort_values(c)
            cols_to_drop = []
            for i in list(self.data):
                if i in target_x_columns:
                    inx = 1
                else:
                    cols_to_drop.append(i)
            if cull == True:
                if sv == 0:
                    start = random.randint(0, len(od.index - (n+6)))
                
            else: 
                n = n
            od = od.iloc[start:start+n+1]
            
            X = od.drop(columns=cols_to_drop)
        y = od[target_y_column]
        X_train = X.iloc[0:test_value]
        y_train = y.iloc[0:test_value]
        X_test = X.iloc[test_value:]
        y_test = y.iloc[test_value:]
       
        return X_train, X_test, y_train, y_test
    

    def get_summary_stats(self):
        """
        This function returns summary statistics for a pandas DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        Returns:
            pd.DataFrame: A DataFrame containing summary statistics.
        """
        return self.data.describe()
    
   
    def filter_data(self):
        # Prompt the user to filter data based on specific criteria
        filter_choice = input("Choose a filter type:\n1. Random PRODUCTLINE filter\n2. Specific PRODUCTLINE(s)\n3. Filter by Status 'Shipped'\n4. Filter by COUNTRY\nEnter choice (1, 2, 3, or 4): ")
        
        if filter_choice == "1":
            # Random PRODUCTLINE filter
            unique_productlines = self.data['PRODUCTLINE'].unique()
            random_productline = np.random.choice(unique_productlines)
            self.data = self.data[self.data['PRODUCTLINE'] == random_productline]
            print(f"Data filtered randomly by PRODUCTLINE: {random_productline}")
        
        elif filter_choice == "2":
            # User-specified PRODUCTLINE filter
            print("Available PRODUCTLINE options:", self.data['PRODUCTLINE'].unique())
            user_selection = input("Enter PRODUCTLINE(s) to filter by (comma-separated if multiple): ").split(',')
            user_selection = [item.strip() for item in user_selection]
            self.data = self.data[self.data['PRODUCTLINE'].isin(user_selection)]
            print(f"Data filtered by user-selected PRODUCTLINE(s): {user_selection}")
        
        elif filter_choice == "3":
            # Filter by Status 'Shipped' for entire dataset 
            self.data = self.data[self.data['STATUS'] == 'Shipped']
            print("Data filtered by Status = 'Shipped'")

        elif filter_choice == "4":
            # Filter by COUNTRY
            print("Available COUNTRY options:", self.data['COUNTRY'].unique())
            selected_country = input("Enter COUNTRY to filter by: ")
            self.data = self.data[self.data['COUNTRY'] == selected_country]
            print(f"Data filtered by COUNTRY: {selected_country}")
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    

    # Need to fix this and incorporate into the actual program...
    def select_and_drop(self, n = 35):
        '''
        Randomly selects n rows from a dataframe and drops the rest

        Args:
            df (pd.DataFrame): the input dataframe
            n (int): the number of rows to randomly select

        Returns:
            pd.DataFrame: a new dataframe filtered to n rows with all other rows dropped
        '''
        # use the sample method to randomly select
        sample_indices = self.data.sample(n=n).index
        self.data = self.data.loc[sample_indices]
        return self.data
    

    def ordered_and_drop(self, n = 35):
        '''
        Randomly selects n rows from a dataframe and drops the rest

        Args:
            df (pd.DataFrame): the input dataframe
            n (int): the number of rows to randomly select

        Returns:
            pd.DataFrame: a new dataframe filtered to n rows with all other rows dropped
        '''
        # use the sample method to randomly select
        start = random.randint(0, len(self.data.index - n-1))
        #print(start)
        df = self.data.iloc[start:start+n]
        self.data = df
        return self.data


    def row_count(self):
        """
        This function returns the number of rows in a pandas DataFrame.
        Args:
            df: The DataFrame to count the rows of.
        Returns:
            The number of rows in the DataFrame.
        """
        return len(self.data.index)


    def aggregate_by_month(self):
        self.data['ORDERDATE'] = pd.to_datetime(self.data['ORDERDATE'])
        start_date = self.data['ORDERDATE'].min()
        self.data['MonthCounter'] = ((self.data['ORDERDATE'].dt.year - start_date.year) * 12 +
                                     self.data['ORDERDATE'].dt.month - start_date.month + 1)
        aggregated_data = self.data.groupby('MonthCounter')['SALES'].sum().reset_index()
        return aggregated_data
    
    
    def aggregate_by_date(self, col1, col2):
        aggregated_data = self.data.groupby('ORDERDATE')[col2].sum().reset_index()
        self.data = aggregated_data
        #print(aggregated_data)
        return aggregated_data



# ModelManager Class ---------------------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
    
        self.model = LinearRegression()
            
       
            

    def fit(self, X, y):
        # feed X train and Y train
        """
        Fit the linear regression model.
        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
        """
        self.model.fit(X, y)


    def predict(self, X):
        """
        Make predictions using the trained model.
        Args:
            X (array-like): The feature matrix for which to make predictions.
        Returns:
            array-like: The predicted values.
        """
        return self.model.predict(X)


    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model's performance.
        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
        Returns:
            tuple: (Mean squared error, R-squared score)
        """
        # LM in this case is y_pred, 'linear model'
        LM_pred = self.predict(y_true)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        return mse, r2, mape


    def get_coefficients(self):
        """
        Get the coefficients of the linear regression model.
        Returns:
            array-like: The coefficients.
        """
        return self.model.coef_


    def get_intercept(self):
        """
        Get the intercept of the linear regression model.
        Returns:
            float: The intercept.
        """
        return self.model.intercept_
    

    def return_weights(self):
        '''
        returns the weights to feed into the scoring function in ScoreManager
        '''
        weights = np.ones(len(self.data))
        return weights

    def auto_reg_lin(self, x, y, X_test):
        #x = np.array(x)
        #y = np.array(y)
        #X_test = np.array(X_test)
        while True:
            try: 
                model = LinearRegression().fit(x, y)
                intercept, coefficients = model.intercept_, model.coef_
                pred = model.predict(X_test) 
            except ValueError:
        #if x is one dimensional
                x = x.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)
                continue
            else:
                break
        return intercept, coefficients, pred

    def poly_fit_pred(self, degree, X, y, x_t):
        """Creates a polynomial regression model of a given degree."""
        
        poly = PolynomialFeatures(degree)
        
        X_poly = poly.fit_transform(X)
        
        self.model.fit(X_poly, y)
        
        Xt_poly = poly.fit_transform(x_t)
        
        
        
        return self.model.predict(Xt_poly)
    

    def mape(self, y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        mape = np.mean(np.abs((y_true - y_pred) / (y_true+.000000001))) * 100
        
        if mape == np.inf:
            mape = 100
        
        elif mape is np.nan:
            mape = 0
            
        elif mape > 100: 
            mape = 100

        
        return mape

# How to Use the ModelManager Class ----------------------
# Create an instance of the class



# Ready for ScoreManager Class !!! ------------------------------------------------------------------------------
class ScoreManager:
    def __init__(self):
        self.score = 0

    
    def scoring(self, pred, actual, weights, ybar):
        #This is the equivalent of the R2 score just out of 100
        score = 0
        #avoid divide by 0
        mean_var = 0.0000000000001
        for i in range(0, len(pred)):
            score += ((pred[i]-actual[i])**2)*weights[i]
            mean_var += ((actual[i]-ybar)**2)*weights[i]
        norm_score = score/mean_var
        grade = (1-norm_score)*100
        grade = round(grade, 3)
        return grade


    def m_scoring(self, pred, actual, weights):
        #mapes scoring, not yet implemented
        score = 0
        #avoid divide by 0
        for i in range(0, len(pred)):
            if actual[i] == 0:
                actual[i] = 0.0000000000001       
            score +=abs((actual[i] - pred[i])/actual[i])*weights[i]
        grade = (score)*100/len(pred)
        grade = round(grade, 3)
        return grade







theme = {
    "background": "#141A2D",
    "text_color": "#FFFFFF",
    "button_bg": "#F65164",
    "button_hover": "#7033FF",
    "accent_color": "#F65164",
    "chart_background": "#252C40",
    "tooltip_bg": "#252C40",
    "tooltip_text_color": "#FFFFFF",
    "neutral": "#DC7653"
}

def bridge(data_set, cull = False):
    
    
    if data_set == "Baby":
        
        #print('babe')
        data_loader_b = DataLoader(babynames)
        
        #print('dl')
        
        #data_loader.handle_missing_values()
        
        X_train, X_test, y_train, y_test = data_loader_b.get_baby()
        
        Plot_Title = f"Baby's born in America each year named {bn}"
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    # this is essentialy hard coded for now but I think making different paths for each dataset based on the collums we actually use makes sense, can be altered later.
    if data_set == "Sales":
        
        data_loader = DataLoader(sales_data)
        
        
        columns_to_drop = ['ORDERNUMBER', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
        categorical_cols = ['STATUS', 'PRODUCTLINE', 'DEALSIZE', 'COUNTRY']
        # add filter productline !!!
        # insert filter for productline here
        data_loader.preprocess_data(columns_to_drop, categorical_cols, 'ORDERDATE')
        data_loader.handle_missing_values()
        #selecting handled by orderd split data
        data_loader.aggregate_by_date('ORDERDATE', 'SALES')
       
        
        

        #data_loader.ordered_and_drop()
        
        X_train, X_test, y_train, y_test = data_loader.ordered_split_data(target_y_column='SALES', target_x_columns=['ORDERDATE'], test_value=-5, cull = cull)

        X_train = X_train.to_numpy()
        X_train = np.transpose(X_train)
        X_train = X_train[0]
        X_test = X_test.to_numpy()
        X_test = np.transpose(X_test)
        X_test = X_test[0]
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        Plot_Title = "Sales by Date"
        
    return  X_train, X_test, y_train, y_test,  Plot_Title


@app.route('/', methods = ['POST', 'GET'])
def start_page():
    #return render_template('start_page.html')
    image_url = 'https://raw.githubusercontent.com/T-Rex-chess/OMSBACapstone/6b4820a52d9294d1ce40371d82244a68d357b879/pics/Xtrapolate.jpg'
    return f'''
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Welcome to Xtrapolate</title>
      <style>
          body {{
              background-color: {theme['background']};
              color: {theme['text_color']};
              font-family: 'Segoe UI', sans-serif;
              margin: 0;
              padding: 20px;
              text-align: center;
          }}
          h1, h2, h3, h4, p {{
              color: {theme['text_color']};
          }}
          hr {{
              border: 1px solid {theme['neutral']};
          }}
          .btn-custom {{
              background-color: {theme['button_bg']};
              color: {theme['text_color']};
              border: none;
              padding: 10px 15px;
              border-radius: 5px;
              font-size: 16px;
              cursor: pointer;
              transition: background-color 0.3s ease;
          }}
          .btn-custom:hover {{
              background-color: {theme['button_hover']};
          }}
          
          
          .collapsible {{
          background-color: {theme['background']};
          color: {theme['text_color']};
          cursor: pointer;
          padding: 20px;
          width: 100%;
          border: none;
          text-align: center;
          outline: none;
          font-size: 15px;
        }}
          
          .collapsible:hover {{
              background-color: {theme['button_hover']};
          }}

          
          .content {{
          padding: 0 500px;
          display: none;
          overflow: hidden;
          background-color: {theme['background']};
        }}
          
          input {{
              background-color: {theme['chart_background']};
              color: {theme['text_color']};
              border: 1px solid {theme['neutral']};
              border-radius: 4px;
              padding: 5px;
              margin: 5px;
          }}
          
      </style>
    </head>
    
    
    <body>
    <img src="{image_url}" alt="Xtrapolate Logo">        
    <h1>Welcome to Xtrapolate!</h1>
    <h2>A Data Science Game</h2>
    <h3>Created by: Thomas Taylor, Jomaica Lei, Andy Turner</h3>
    <hr> </hr>
    <h2>Think you can beat the Machine? </h2>
    <p> In this game, you are given a graph of past values from various datasets and asked to predict the next 5 datapoints.  </p>
    <p> Watch out, because a machine learning model will also be trying to guess. </p>
    <p> The sales dataset is sourced from Kaggle, and is available here: 
            <a href="https://www.kaggle.com/datasets/kyanyoga/sample-sales-data" style="color: {theme['accent_color']}; text-decoration: none;">Sales Dataset</a>
    </p>
    
    <p> The Baby Names Data comes from the US Social Security Administration, and is available here: 
            <a href="https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data" style="color: {theme['accent_color']}; text-decoration: none;">Baby Names Dataset</a>
    </p>
    
    <hr> </hr>
    <button type="button" class="collapsible">How to Play </button>
    <div class="content">
        <p> The game will begin by displaying a scatterplot of some sales data. The scatterplot represents the
                total sales (price x quantity) of vehicles sold globally across various regions.
                You will be prompted to enter guesses on the total sales of vehicles for 5 specific dates. 
                A machine learning model will also be running to predict the sales as well. 
                Your job is to do a better job of predicting than the machine.</p>
    </div>

    <hr> </hr>

    <button type="button" class="collapsible">Game Scoring</button>
    <div class="content">

        <p> The game is scored using Mean Absolute Percentage Error 
        <a href="https://en.wikipedia.org/wiki/Mean_absolute_percentage_error" style="color: {theme['accent_color']}; text-decoration: none;">(MAPE)</a>. 
            MAPE is a statistical measure that calculates the average percentage difference 
            between predicted values and actual values. This essentially shows how far off a model's predictions are on average.
            MAPE is expressed as a percentage, making it easy to interpret the accuracy of a forecast or prediction.
            Lastly, a lower MAPE indicates a more accurate model.</p>
    </div>

    <hr> </hr>
    
    <form action="/guess" method = "POST">
  <label for="dta">Choose a dataset and go:</label>
  <select name="dta" id="dta" >
    <option value="random">Random</option>
    <option value="Baby">Baby Names</option>
    <option value="Sales">Sales</option>

  </select>

  <input type = "submit" value = "Confirm">
  </form>
  
    
    <script>
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {{
  coll[i].addEventListener("click", function() {{
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {{
      content.style.display = "none";
    }} else {{
      content.style.display = "block";
    }}
  }});
}}
</script>
    
    
   
    
    
    </body>
    </html>
    '''
   

# NEED TO FEED AN ARRAY INTO THE GUESS FUNCTION SO IT WORKS
@app.route('/guess/', methods = ['POST', 'GET'])
def guess():
    

    
    if request.method == 'GET':
        return f"The URL /guess is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        # Creating Plot Figure
        
        #print('baby')
        
        
        
        
        global selected_data_set
        
        selected_data_set = request.form.get("dta")
        
        if selected_data_set == 'random':
            
            selected_data_set = random.choice(['Baby', 'Sales'])
        
        
        global counter
        global sv
        if sv == 0:
            counter += 1
        
        
        
        if selected_data_set == 'Baby':
            
            X_train, X_test, y_train, y_test,  Plot_Title = bridge(selected_data_set, True)

            start_shade = X_test[0]
            end_shade = X_test[4]
            box = BoxAnnotation(left=start_shade, right=end_shade, fill_alpha=0.4, fill_color='lightblue')
            
            p = figure(height=350, sizing_mode="stretch_width")
            #for changing chart background color: , background_fill_color=theme['chart_background']
            p.xaxis.axis_label = "Year"
            p.yaxis.axis_label = f"Number of Babies named {bn}"
            #p.add_tools(HoverTool())
            # Defining Plot to be a Scatter Plot
            p.scatter( 	[i for i in X_train],
        		[j for j in y_train],
                size=20,
                color="black",
                alpha=0.5
            )
            
    
            #p.legend.location = 'top_left'
            p.add_layout(box)
            
            # Get Chart Components
            script, div = components(p)
        
            
            
            sv = 1
            # Return the components to the HTML template
            return f'''
        	<html lang="en">
        		<head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
        			<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js"></script>
        			<title>Bokeh Charts</title>
                    <style>
                        body {{
                            background-color: {theme['background']};
                            color: {theme['text_color']};
                            font-family: 'Segoe UI', sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: left;
                        }}
                        h1, h2, h3, p {{
                            color: {theme['text_color']};
                        }}
                        .btn-custom {{
                            background-color: {theme['button_bg']};
                            color: {theme['text_color']};
                            border: none;
                            padding: 10px 15px;
                            border-radius: 5px;
                            font-size: 16px;
                            cursor: pointer;
                            transition: background-color 0.3s ease;
                        }}
                        .btn-custom:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        input {{
                            background-color: {theme['chart_background']};
                            color: {theme['text_color']};
                            border: 1px solid {theme['neutral']};
                            border-radius: 4px;
                            padding: 5px;
                            margin: 5px;
                        }}
                        
                        
                        section.inputSection {{
                            border-style: solid;
                            /* background: linear-gradient(60deg, {theme['background']}, {theme['button_hover']}); */
                            border-color: {theme['button_bg']};
                            margin: 10px;
                            text-align: left;
                        }}
                                                           
                       .guessstyle {{
        border-style: solid;
        background: linear-gradient(60deg,{theme['background']}, {theme['button_hover']});
        max-width: fit-content;
        border-color: black;
        position: relative;
        padding: 10;
        
    }}
                            .bk-Figure {{
        max-width: 80%;
        
    }}
                    </style>       
                
                </head>
                
        		<body>
        			<h1> {Plot_Title} </h1>
                    
        			{ div }
        			{ script }
    
                    <h3> Round {str(counter)}: Submit a prediction for the number of babies born named {bn} in each of the below 5 years. This will show up in the light blue shaded region. </h3>
                    
                    <section class="inputSection">
                    
                    <p> After you submit your guesses, a machine learning model will also make some predictions. Can you beat the machine by predicting values more accurately? Good luck! </p>
                    
                    <form action="/display" method = "POST" class = "guessstyle">
            <p> {bn}'s born in {str(X_test[0]):.10} <input type = "number" step = "any" name = "g1" value = 0 required /></p>
            <p> {bn}'s born in {str(X_test[1]):.10} <input type = "number" step = "any" name = "g2" value = 0  required /></p>
            <p> {bn}'s born in {str(X_test[2]):.10} <input type = "number" step = "any" name = "g3" value = 0 required /></p>
            <p> {bn}'s born in {str(X_test[3]):.10} <input type = "number" step = "any" name = "g4" value = 0 required /></p>
            <p> {bn}'s born in {str(X_test[4]):.10} <input type = "number" step = "any" name = "g5" value = 0 required /></p>
            <p><input type = "submit" value = "Submit" /></p>
            </form>
                    </section>
        		</body>
        	</html> '''
        if selected_data_set == 'Sales':
            X_train, X_test, y_train, y_test,  Plot_Title = bridge(selected_data_set, True)
            start_shade = X_test[0]
            end_shade = X_test[4]
            box = BoxAnnotation(left=start_shade, right=end_shade, fill_alpha=0.4, fill_color='lightblue')
            
            p = figure(height=350, x_axis_type='datetime', sizing_mode="stretch_width")
            #for changing chart background color: , background_fill_color=theme['chart_background']
            p.xaxis.axis_label = "Calendar Date"
            p.yaxis.axis_label = "Sum of Vehicle Sales"
            p.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
            # p.add_tools(HoverTool())
            # Defining Plot to be a Scatter Plot
            p.scatter( 	[i for i in X_train],
        		[j for j in y_train],
                size=20,
                color="black",
                alpha=0.5
            )
            #p.legend.location = 'top_left'
            p.add_layout(box)
            
            # Get Chart Components
            script, div = components(p)
        

            
            sv = 1
            # Return the components to the HTML template
            return f'''
    	<html lang="en">
    		<head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
    			<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js"></script>
    			<title>Bokeh Charts</title>
                <style>
                    body {{
                        background-color: {theme['background']};
                        color: {theme['text_color']};
                        font-family: 'Segoe UI', sans-serif;
                        margin: 0;
                        padding: 20px;
                        text-align: left;
                    }}
                    h1, h2, h3, p {{
                        color: {theme['text_color']};
                    }}
                    .btn-custom {{
                        background-color: {theme['button_bg']};
                        color: {theme['text_color']};
                        border: none;
                        padding: 10px 15px;
                        border-radius: 5px;
                        font-size: 16px;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }}
                    .btn-custom:hover {{
                        background-color: {theme['button_hover']};
                    }}
                    input {{
                        background-color: {theme['chart_background']};
                        color: {theme['text_color']};
                        border: 1px solid {theme['neutral']};
                        border-radius: 4px;
                        padding: 5px;
                        margin: 5px;
                    }}
                    
                    
                    section.inputSection {{
                        border-style: solid;
                        /* background: linear-gradient(60deg, {theme['background']}, {theme['button_hover']}); */
                        border-color: {theme['button_bg']};
                        margin: 10px;
                        text-align: left;
                    }}
                                                       
                   .guessstyle {{
    border-style: solid;
    background: linear-gradient(60deg,{theme['background']}, {theme['button_hover']});
    max-width: fit-content;
    border-color: black;
    position: relative;
    padding: 10;
    
}}
                        .bk-Figure {{
    max-width: 80%;
    
}}
                </style>           
            
            </head>
    		<body>
    			<h1> Graph of Sum of Vehicle Sales ($) by Calendar Date: Round {str(counter)} </h1>
                <h3> The below scatterplot displays the aggregate sum of the $ amount of vehicles sold on a given calendar date (Sales = Price * Quantity Sold) </h3>
    			{ div }
    			{ script }

                <h3> Submit a prediction for the $ amount of vehicle sales for each date below. This will show up in the light blue shaded region. </h3>
                <section class="inputSection">
                <p> After you submit your guesses, a machine learning model will also make some predictions. Can you beat the machine by predicting values more accurately? Good luck! </p>
                
                <form action="/display" method = "POST" class = "guessstyle">
        <p> Vehicle Sales on {str(X_test[0]):.10} <input type = "number" step = "any" name = "g1" value = 0 required /></p>
        <p> Vehicle Sales on {str(X_test[1]):.10} <input type = "number" step = "any" name = "g2" value = 0  required /></p>
        <p> Vehicle Sales on {str(X_test[2]):.10} <input type = "number" step = "any" name = "g3" value = 0 required /></p>
        <p> Vehicle Sales on {str(X_test[3]):.10} <input type = "number" step = "any" name = "g4" value = 0 required /></p>
        <p> Vehicle Sales on {str(X_test[4]):.10} <input type = "number" step = "any" name = "g5" value = 0 required /></p>
        <p><input type = "submit" value = "Submit" /></p>
        </form>
                </section>
    		</body>
    	</html>
    	'''
    	



@app.route('/display/', methods = ['POST', 'GET'])
def display():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        
        print('h')
        
                
        global selected_data_set
        
        global sv
        global counter
        sv = 1
        
       
        print(selected_data_set)
        #weights = np.ones(len(y_test))
        #ybar = (sum(y_train)+sum(y_test))/(len(y_train)+len(y_test))
        
        #create instance of ScoreManager()
        #s = ScoreManager()
        #user_score = s.scoring(user_guesses, y_test, weights, ybar)
        
        #create instance of ModelManager()
        if selected_data_set == 'Baby':
            
            X_train, X_test, y_train, y_test,  Plot_Title = bridge(selected_data_set)
            
            '''
            train_days = []
            test_days = []
            for i in X_test:
                delt = i - X_train[0]
                d =  delt.astype('timedelta64[D]')
                test_days.append(d / np.timedelta64(1, 'D'))
            for i in X_train:
                delt = i - X_train[0]
                d =  delt.astype('timedelta64[D]')
                train_days.append(d / np.timedelta64(1, 'D'))
            '''
            

            # user guesses
            user_guesses = []
            user_guesses.append(float(request.form.get("g1")))
            user_guesses.append(float(request.form.get("g2")))
            user_guesses.append(float(request.form.get("g3")))
            user_guesses.append(float(request.form.get("g4")))
            user_guesses.append(float(request.form.get("g5")))
            
            regr = ModelManager()
            
            #ntercept, coefficients, ML_pred = regr.auto_reg_lin(X_train, y_train, X_test)
            ML_pred  = regr.poly_fit_pred(2, X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1))
            #gamecoefficients = regr.get_coefficients()
            #intercepts = regr.get_intercept()
            #ML_pred = regr.predict(X_test.reshape(-1, 1))
            #ML_Score = s.scoring(ML_pred, y_test, weights, ybar)
    
            #get user MAPE:
            user_mape = round(regr.mape(y_test, user_guesses),2)
    
            #get ML MAPE:
            regr_mape = round(regr.mape(y_test, ML_pred),2)
    
            #create plot
            p = figure(height=350,  sizing_mode="stretch_width")
            p.xaxis.axis_label = "Year"
            p.yaxis.axis_label = f"Number of Babies named {bn}"
            #p.add_tools(HoverTool())
    
            # Defining Plot to be a Scatter Plot
            p.scatter(
                [i for i in X_train],
                [j for j in y_train],
                size=20,
                color="black",
                alpha=0.5,
                legend_label = "Historical Actuals"
            )
            
            p.scatter(
                [i for i in X_test],
                [j for j in y_test],
                size=20,
                color="purple",
                alpha=0.5,
                legend_label = 'Actual Values (from Prediction Dates)'
            )
            
            p.scatter(
                [i for i in X_test],
                [j for j in user_guesses],
                size=20,
                color="teal",
                alpha=0.5,
                legend_label = "Player Predicted value"
            )
            
            p.scatter(
                [i for i in X_test],
                [j for j in ML_pred],
                size=20,
                color="blue",
                alpha=0.5,
                legend_label = "ML Predicted value"
            )
            
            p.legend.location = 'top_left'
    
            # Get Chart Components
            script, div = components(p)
            
            sv = 0
            
            if user_mape <= regr_mape: 
                return  f'''
            <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js"></script>
                    <title>Bokeh Charts 2</title>
                    <style>
                        .bk-Figure {{
                                max-width: 60%;
                                left: 20%;
                            }}
                        body {{
                            background-color: {theme['background']};
                            color: {theme['text_color']};
                            font-family: 'Segoe UI', sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                        }}
                        h1, h2, h3, p {{
                            color: {theme['text_color']};
                        }}
                        .btn-custom {{
                            background-color: {theme['button_bg']};
                            color: {theme['text_color']};
                            border: none;
                            padding: 10px 15px;
                            border-radius: 5px;
                            font-size: 16px;
                            cursor: pointer;
                            transition: background-color 0.3s ease;
                        }}
                        .btn-custom:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        input {{
                            background-color: {theme['chart_background']};
                            color: {theme['text_color']};
                            border: 1px solid {theme['neutral']};
                            border-radius: 4px;
                            padding: 5px;
                            margin: 5px;
                        }}
                        
    
                        
                        .collapsible {{
                        background-color: {theme['background']};
                        color: {theme['text_color']};
                        cursor: pointer;
                        padding: 20px;
                        width: 10%;
                        border: 1px solid {theme['neutral']};
                        border-radius: 4px;
                        text-align: center;
                        outline: none;
                        font-size: 25px;
                      }}
                        
                        .collapsible:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        
    
                        
                        .content {{
                        padding: 0 500px;
                        display: none;
                        overflow: hidden;
                        text-align: center;
                        background-color: {theme['background']};
                      }}
                        table, th, td {{
                      border: 2px solid red;
                      border-collapse: collapse;
                      text-align: center;
                    }}
                    th, td {{
                      padding: 10px;
                    }}
                    
                    .center {{
                      margin-left: auto;
                      margin-right: auto;
                    }}
                        
                    </style>     
                    
                </head>
                <body>
                    <h1> Here are the results of the predictions: </h1>
                    { div }
                    { script }
                    <h2> You Won Round {str(counter)}! Good job predicting! </h2>
                    <h3> Your MAPE was: {user_mape}, ML's MAPE Was {regr_mape} <h3>
                    <p> Game is scored using Mean Absolute Percentage Error (MAPE). Higher MAPE = Less Accurate, Lower MAPE = More Accurate </p>
                    
                    
                    <button type="button" class="collapsible"> Results Table </button>
                    <div class="content">
                        <table class="center">
                          <tr>
                            <th>Year</th>
                            <th>Actual Value</th>
                            <th>User Prediction</th>
                            <th>ML Prediction</th>
                          </tr>
                          <tr>
                            <td>{str(X_test[0]):.10}</td>
                            <td>{y_test[0]}</td>
                            <td>{user_guesses[0]}</td>
                            <td>{float(int(ML_pred[0]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[1]):.10}</td>
                            <td>{y_test[1]}</td>
                            <td>{user_guesses[1]}</td>
                            <td>{float(int(ML_pred[1]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[2]):.10}</td>
                            <td>{y_test[2]}</td>
                            <td>{user_guesses[2]}</td>
                            <td>{float(int(ML_pred[2]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[3]):.10}</td>
                            <td>{y_test[3]}</td>
                            <td>{user_guesses[3]}</td>
                           <td>{float(int(ML_pred[3]))}</td>
                        
                         <tr>
                          <tr>
                            <td>{str(X_test[4]):.10}</td>
                            <td>{y_test[4]}</td>
                            <td>{user_guesses[4]}</td>
                            <td>{float(int(ML_pred[4]))}</td>
                         <tr>
                        
                        </table>
                    </div>
                    
                    
                    <form action="/guess" method = "POST">
            <p><input type = "submit" value = "Nice work, think you can do it again?" /></p>
            </form>
                    <form action="/" method = "POST">
            <p><input type = "submit" value = "Return to Homepage" /></p>
            </form>
            
            <script>
        var coll = document.getElementsByClassName("collapsible");
        var i;
    
        for (i = 0; i < coll.length; i++) {{
          coll[i].addEventListener("click", function() {{
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.display === "block") {{
              content.style.display = "none";
            }} else {{
              content.style.display = "block";
            }}
          }});
        }}
        </script>
            
                </body>
            </html>
            '''
            else:
                return  f'''
            <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js"></script>
                    <title>Bokeh Charts 2</title>
                    <style>
                        .bk-Figure {{
                            max-width: 60%;
                            left: 20%;
                        }}
                        body {{
                            background-color: {theme['background']};
                            color: {theme['text_color']};
                            font-family: 'Segoe UI', sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                        }}
                        h1, h2, h3, p {{
                            color: {theme['text_color']};
                        }}
                        .btn-custom {{
                            background-color: {theme['button_bg']};
                            color: {theme['text_color']};
                            border: none;
                            padding: 10px 15px;
                            border-radius: 5px;
                            font-size: 16px;
                            cursor: pointer;
                            transition: background-color 0.3s ease;
                        }}
                        .btn-custom:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        input {{
                            background-color: {theme['chart_background']};
                            color: {theme['text_color']};
                            border: 1px solid {theme['neutral']};
                            border-radius: 4px;
                            padding: 5px;
                            margin: 5px;
                        }}
                        
                        
                        .collapsible {{
                        background-color: {theme['background']};
                        color: {theme['text_color']};
                        cursor: pointer;
                        padding: 20px;
                        width: 10%;
                        border: 1px solid {theme['neutral']};
                        border-radius: 4px;
                        text-align: center;
                        outline: none;
                        font-size: 25px;
                      }}
                        
                        .collapsible:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        
    
    
                        
                        .content {{
                        padding: 0 500px;
                        display: none;
                        overflow: hidden;
                        background-color: {theme['background']};
                      }}
                        
                        table, th, td {{
                      border: 2px solid red;
                      border-collapse: collapse;
                    }}
                    th, td {{
                      padding: 10px;
                    }}
                        
                    </style>       
                    
                </head>
                <body>
                    <h1> Here are the results of the predictions: </h1>
                    { div }
                    { script }
                <h2> ML Wins Round {str(counter)}! Better luck next time. </h2>
                <h3> Your MAPE was: {user_mape}, ML's MAPE Was {regr_mape} <h3>                    
                <p> Game is scored using Mean Absolute Percentage Error (MAPE). Higher MAPE = Less Accurate, Lower MAPE = More Accurate </p>
            <button type="button" class="collapsible"> Results Table </button>
                <div class="content">
                        <table>
                          <tr>
                            <th>Year</th>
                            <th>Actual Value</th>
                            <th>User Prediction</th>
                            <th>ML Prediction</th>
                          </tr>
                          <tr>
                            <td>{str(X_test[0]):.10}</td>
                            <td>{y_test[0]}</td>
                            <td>{user_guesses[0]}</td>
                            <td>{float(int(ML_pred[0]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[1]):.10}</td>
                            <td>{y_test[1]}</td>
                            <td>{user_guesses[1]}</td>
                            <td>{float(int(ML_pred[1]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[2]):.10}</td>
                            <td>{y_test[2]}</td>
                            <td>{user_guesses[2]}</td>
                            <td>{float(int(ML_pred[2]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[3]):.10}</td>
                            <td>{y_test[3]}</td>
                            <td>{user_guesses[3]}</td>
                           <td>{float(int(ML_pred[3]))}</td>
                        
                         <tr>
                          <tr>
                            <td>{str(X_test[4]):.10}</td>
                            <td>{y_test[4]}</td>
                            <td>{user_guesses[4]}</td>
                            <td>{float(int(ML_pred[4]))}</td>
                         <tr>
                        
                        </table>
                </div>
                
                
                <form action="/guess" method = "POST">
        <p><input type = "submit" value = "Nice work, think you can do it again?" /></p>
        </form>
                <form action="/" method = "POST">
        <p><input type = "submit" value = "Return to Homepage" /></p>
        </form>
        
        <script>
    var coll = document.getElementsByClassName("collapsible");
    var i;
    
    for (i = 0; i < coll.length; i++) {{
      coll[i].addEventListener("click", function() {{
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {{
          content.style.display = "none";
        }} else {{
          content.style.display = "block";
        }}
      }});
    }}
    </script>
        
            </body>
        </html>'''
        if selected_data_set == 'Sales':
            

            
            
            X_train, X_test, y_train, y_test,  Plot_Title = bridge(selected_data_set)
            
            train_days = []
            test_days = []
            for i in X_test:
               delt = i - X_train[0]
               d =  delt.astype('timedelta64[D]')
               test_days.append(d / np.timedelta64(1, 'D'))
            for i in X_train:
               delt = i - X_train[0]
               d =  delt.astype('timedelta64[D]')
               train_days.append(d / np.timedelta64(1, 'D'))
           
           
            train_days = np.array(train_days)
            test_days = np.array(test_days)
           # user guesses
            user_guesses = []
            user_guesses.append(float(request.form.get("g1")))
            user_guesses.append(float(request.form.get("g2")))
            user_guesses.append(float(request.form.get("g3")))
            user_guesses.append(float(request.form.get("g4")))
            user_guesses.append(float(request.form.get("g5")))
           
            weights = np.ones(len(y_test))
            ybar = (sum(y_train)+sum(y_test))/(len(y_train)+len(y_test))
           
            
            
           #create instance of ScoreManager()
            s = ScoreManager()
            user_score = s.scoring(user_guesses, y_test, weights, ybar)
                
            regr = ModelManager()
        
            #ntercept, coefficients, ML_pred = regr.auto_reg_lin(X_train, y_train, X_test)
            regr.fit(train_days.reshape(-1, 1), y_train)
            gamecoefficients = regr.get_coefficients()
            intercepts = regr.get_intercept()
            ML_pred = regr.predict(test_days.reshape(-1, 1))
            ML_Score = s.scoring(ML_pred, y_test, weights, ybar)
    
            #get user MAPE:
            user_mape = round(regr.mape(y_test, user_guesses),2)
            
            #get ML MAPE:
            regr_mape = round(regr.mape(y_test, ML_pred),2)
    
            #create plot
            p = figure(height=350, x_axis_type='datetime', sizing_mode="stretch_width")
            p.xaxis.axis_label = "Calendar Date"
            p.yaxis.axis_label = "Sum of Vehicle Sales"
            p.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
            # p.add_tools(HoverTool())
    
            # Defining Plot to be a Scatter Plot
            p.scatter(
                [i for i in X_train],
                [j for j in y_train],
                size=20,
                color="black",
                alpha=0.5,
                legend_label = "Historical Actuals"
            )
            
            p.scatter(
                [i for i in X_test],
                [j for j in y_test],
                size=20,
                color="purple",
                alpha=0.5,
                legend_label = 'Actual Values (from Prediction Dates)'
            )
            
            p.scatter(
                [i for i in X_test],
                [j for j in user_guesses],
                size=20,
                color="teal",
                alpha=0.5,
                legend_label = "Player Predicted value"
            )
            
            p.scatter(
                [i for i in X_test],
                [j for j in ML_pred],
                size=20,
                color="blue",
                alpha=0.5,
                legend_label = "ML Predicted value"
            )
            
            p.legend.location = 'top_left'
    
            # Get Chart Components
            script, div = components(p)
            
            sv = 0
            
           
            
            if user_mape < regr_mape:
                
                return  f'''
            <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js"></script>
                    <title>Bokeh Charts 2</title>
                    <style>
                        .bk-Figure {{
                                max-width: 60%;
                                left: 20%;
                            }}
                        body {{
                            background-color: {theme['background']};
                            color: {theme['text_color']};
                            font-family: 'Segoe UI', sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                        }}
                        h1, h2, h3, p {{
                            color: {theme['text_color']};
                        }}
                        .btn-custom {{
                            background-color: {theme['button_bg']};
                            color: {theme['text_color']};
                            border: none;
                            padding: 10px 15px;
                            border-radius: 5px;
                            font-size: 16px;
                            cursor: pointer;
                            transition: background-color 0.3s ease;
                        }}
                        .btn-custom:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        input {{
                            background-color: {theme['chart_background']};
                            color: {theme['text_color']};
                            border: 1px solid {theme['neutral']};
                            border-radius: 4px;
                            padding: 5px;
                            margin: 5px;
                        }}
                        
    
                        
                        .collapsible {{
                        background-color: {theme['background']};
                        color: {theme['text_color']};
                        cursor: pointer;
                        padding: 20px;
                        width: 10%;
                        border: 1px solid {theme['neutral']};
                        border-radius: 4px;
                        text-align: center;
                        outline: none;
                        font-size: 25px;
                      }}
                        
                        .collapsible:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        
    
                        
                        .content {{
                        padding: 0 500px;
                        display: none;
                        overflow: hidden;
                        text-align: center;
                        background-color: {theme['background']};
                      }}
                        table, th, td {{
                      border: 2px solid red;
                      border-collapse: collapse;
                      text-align: center;
                    }}
                    th, td {{
                      padding: 10px;
                    }}
                    
                    .center {{
                      margin-left: auto;
                      margin-right: auto;
                    }}
                        
                    </style>     
                    
                </head>
                <body>
                    <h1> Here are the results of the predictions for this round: </h1>
                    { div }
                    { script }
                    <h3> You Won this Round! Good job predicting! </h3>
                    <h3> Your MAPE was: {user_mape}, ML's MAPE Was {regr_mape} <h3>
                    <p> Game is scored using Mean Absolute Percentage Error (MAPE). Higher MAPE = Less Accurate, Lower MAPE = More Accurate </p>
                    
                    <button type="button" class="collapsible"> Results Table </button>
                        <div class="content">
                                <table>
                                  <tr>
                                    <th>Date</th>
                                    <th>Actual Value</th>
                                    <th>User Prediction</th>
                                    <th>ML Prediction</th>
                                  </tr>
                                  <tr>
                                    <td>{str(X_test[0]):.10}</td>
                                    <td>{float(int(y_test[0]))}</td>
                                    <td>{user_guesses[0]}</td>
                                    <td>{float(int(ML_pred[0]))}</td>
                                 <tr>
                                  <tr>
                                    <td>{str(X_test[1]):.10}</td>
                                    <td>{float(int(y_test[1]))}</td>
                                    <td>{user_guesses[1]}</td>
                                    <td>{float(int(ML_pred[1]))}</td>
                                 <tr>
                                  <tr>
                                    <td>{str(X_test[2]):.10}</td>
                                    <td>{float(int(y_test[2]))}</td>
                                    <td>{user_guesses[2]}</td>
                                    <td>{float(int(ML_pred[2]))}</td>
                                 <tr>
                                  <tr>
                                    <td>{str(X_test[3]):.10}</td>
                                    <td>{float(int(y_test[3]))}</td>
                                    <td>{user_guesses[3]}</td>
                                   <td>{float(int(ML_pred[3]))}</td>
                                
                                 <tr>
                                  <tr>
                                    <td>{str(X_test[4]):.10}</td>
                                    <td>{float(int(y_test[4]))}</td>
                                    <td>{user_guesses[4]}</td>
                                    <td>{float(int(ML_pred[4]))}</td>
                                 <tr>
                                
                                </table>
                        </div>
                    
                    
                    <form action="/guess" method = "POST">
            <p><input type = "submit" value = "Play Again" /></p>
            </form>
                    <form action="/" method = "POST">
            <p><input type = "submit" value = "Return to Homepage" /></p>
            </form>
            
            <script>
        var coll = document.getElementsByClassName("collapsible");
        var i;
        
        for (i = 0; i < coll.length; i++) {{
          coll[i].addEventListener("click", function() {{
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.display === "block") {{
              content.style.display = "none";
            }} else {{
              content.style.display = "block";
            }}
          }});
        }}
        </script>
            
                </body>
            </html>
            '''
            else:
                return  f'''
            <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js"></script>
                    <title>Bokeh Charts 2</title>
                    <style>
                        .bk-Figure {{
                                max-width: 60%;
                                left: 20%;
                            }}
                        body {{
                            background-color: {theme['background']};
                            color: {theme['text_color']};
                            font-family: 'Segoe UI', sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                        }}
                        h1, h2, h3, p {{
                            color: {theme['text_color']};
                        }}
                        .btn-custom {{
                            background-color: {theme['button_bg']};
                            color: {theme['text_color']};
                            border: none;
                            padding: 10px 15px;
                            border-radius: 5px;
                            font-size: 16px;
                            cursor: pointer;
                            transition: background-color 0.3s ease;
                        }}
                        .btn-custom:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        input {{
                            background-color: {theme['chart_background']};
                            color: {theme['text_color']};
                            border: 1px solid {theme['neutral']};
                            border-radius: 4px;
                            padding: 5px;
                            margin: 5px;
                        }}
                        
    
                        
                        .collapsible {{
                        background-color: {theme['background']};
                        color: {theme['text_color']};
                        cursor: pointer;
                        padding: 20px;
                        width: 10%;
                        border: 1px solid {theme['neutral']};
                        border-radius: 4px;
                        text-align: center;
                        outline: none;
                        font-size: 25px;
                      }}
                        
                        .collapsible:hover {{
                            background-color: {theme['button_hover']};
                        }}
                        
    
                        
                        .content {{
                        padding: 0 500px;
                        display: none;
                        overflow: hidden;
                        text-align: center;
                        background-color: {theme['background']};
                      }}
                        table, th, td {{
                      border: 2px solid red;
                      border-collapse: collapse;
                      text-align: center;
                    }}
                    th, td {{
                      padding: 10px;
                    }}
                    
                    .center {{
                      margin-left: auto;
                      margin-right: auto;
                    }}
                        
                    </style>       
                    
                </head>
                <body>
                    <h1> Here are the results of the predictions: </h1>
                    { div }
                    { script }
                <h2> ML Wins this Round! Better luck next time. </h2>
                <h3> Your MAPE was: {user_mape}, ML's MAPE Was {regr_mape} <h3>                    
                <p> Game is scored using Mean Absolute Percentage Error (MAPE). Higher MAPE = Less Accurate, Lower MAPE = More Accurate </p>
           
            <button type="button" class="collapsible"> Results Table </button>
                <div class="content">
                        <table>
                          <tr>
                            <th>Date</th>
                            <th>Actual Value</th>
                            <th>User Prediction</th>
                            <th>ML Prediction</th>
                          </tr>
                          <tr>
                            <td>{str(X_test[0]):.10}</td>
                            <td>{float(int(y_test[0]))}</td>
                            <td>{user_guesses[0]}</td>
                            <td>{float(int(ML_pred[0]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[1]):.10}</td>
                            <td>{float(int(y_test[1]))}</td>
                            <td>{user_guesses[1]}</td>
                            <td>{float(int(ML_pred[1]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[2]):.10}</td>
                            <td>{float(int(y_test[2]))}</td>
                            <td>{user_guesses[2]}</td>
                            <td>{float(int(ML_pred[2]))}</td>
                         <tr>
                          <tr>
                            <td>{str(X_test[3]):.10}</td>
                            <td>{float(int(y_test[3]))}</td>
                            <td>{user_guesses[3]}</td>
                           <td>{float(int(ML_pred[3]))}</td>
                        
                         <tr>
                          <tr>
                            <td>{str(X_test[4]):.10}</td>
                            <td>{float(int(y_test[4]))}</td>
                            <td>{user_guesses[4]}</td>
                            <td>{float(int(ML_pred[4]))}</td>
                         <tr>
                        
                        </table>
                </div>
           
            <form action="/guess" method = "POST">
    <p><input type = "submit" value = "Play Again" /></p>
        </form>
            <form action="/" method = "POST">
    <p><input type = "submit" value = "Return to Homepage" /></p>
    </form>
    
            <script>
            var coll = document.getElementsByClassName("collapsible");
            var i;
            
            for (i = 0; i < coll.length; i++) {{
              coll[i].addEventListener("click", function() {{
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {{
                  content.style.display = "none";
                }} else {{
                  content.style.display = "block";
                }}
              }});
            }}
            </script>
                </body>
            </html>
            '''
        
#X_train, X_test, y_train, y_test,  Plot_Title = bridge("Sales") 
#app.run(debug=False)

if __name__ == '__main__':
	# Run the application on the local development server
	app.run(debug=False)
