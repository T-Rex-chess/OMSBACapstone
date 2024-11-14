
# Imports -----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

from datetime import datetime
from datetime import timedelta
#Xtrapolate Functions
import random

#Bokeh Charts.py
from flask import Flask
from flask import Flask, render_template, request
from bokeh.embed import components
from bokeh.plotting import figure
# from xtrapolate_functions import scoring, auto_reg_lin
from bokeh.models import HoverTool


# Flask constructor ------------------------------------------------------------------------------------------------------
app = Flask(__name__)


# Arrays -----------------------------------------------------------------------------------------------------------------
# Predefined dataset of topics
topics = [
    "Sales",
    "Electric Vehicles",
    "Baby Names",
    "Break Function",

]

#  "Electric Vehicles",
#  "Baby Names",

difficulty = [
    "Easy",
    "Medium",
    "Hard",
    "Break Function",
]

status = [
    '1',
    '2',
]


# Connect to Data --------------------------------------------------------------------------------------------------------
file_url = 'https://raw.githubusercontent.com/T-Rex-chess/OMSBACapstone/main/data/sales_data_sample.csv'
sales_data = pd.read_csv(file_url, encoding='ISO-8859-1') #Handles wider range of special char.
sales_data_df = pd.DataFrame(sales_data) # create a dataframe of the sales data
#print(sales_data.head())
#print("\n Sales Dataframe Head:")
#print(sales_data_df.head())



# BaseManager Parent Class -----------------------------------------------------------------------------------------------
# BaseManager class: Parent class.
class BaseManager:
    def __init__(self, data):
        self.data = data



# GameManager Parent Class ------------------------------------------------------------------------------------------------
class GameManager:
    def __init__(self, topics, difficulty, game_counter, game_status):
        self.topics = topics
        self.difficulty = difficulty
        self.game_counter = 0
        self.game_status = 1

    def display_topics(self):
        print("Available topics:")
        for i, topic in enumerate(self.topics, start=1):
            print(f"{i}. {topic}")

    def get_topic_selection(self):
        while True:
            try:
                choice = int(input("Please select a topic by entering the corresponding number: "))
                if choice == 4:
                    break
                if 1 <= choice <= len(self.topics):
                    return self.topics[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(self.topics)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def display_difficulty(self):
        print("Available Difficulties:")
        for i, difficulty in enumerate(self.difficulty, start=1):
            print(f"{i}. {difficulty}")

    def get_difficulty_selection(self):
        while True:
            try:
                choice = int(input("Please select a difficulty by entering the corresponding number: "))
                if choice == 4:
                    break
                if 1 <= choice <= len(self.difficulty):
                    return difficulty[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(self.difficulty)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Asks the player if they want to continue playing
    def update_game_status(self):
        while True:
            try:
                choice = int(input("Would you like to play again? Enter 1 for Yes, 2 for No: "))
                if choice == 1:
                    return self.game_status
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")


    # Create a gameplay loop counter (ie a round) that increments after player has completed first round
    def increment_game_counter(self):
        while self.game_status == 1: 
            print('Completed Round \n', self.game_counter)
            self.game_counter += 1 
            print('Would you like to play again?')
    

''' # defunct with html form in gui
    def get_guess(self, YLab):
        guess_list = []
        for i in range(5):
            g = askfloat("Input", f"Enter your guess for {YLab} {i}")
            while g == None:
                g = askfloat("Input", f"Enter your guess for {YLab} {i}")
            guess_list.append(g)
        return guess_list
'''



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

    
    def preprocess_data(self):
        # Drop irrelevant columns
        columns_to_drop = ['ORDERNUMBER', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
        self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        # Encode categorical variables using one-hot encoding
        categorical_cols = ['STATUS', 'PRODUCTLINE', 'DEALSIZE', 'COUNTRY']
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        # Drop non-numeric columns after encoding
        
        #convert dates to datetime
        self.data['ORDERDATE'] = pd.to_datetime(self.data['ORDERDATE'])
        
        #self.data = self.data.select_dtypes(include=[np.number, pd._libs.tslibs.timestamps.Timestamp])
        print("Non-numeric columns dropped and categorical variables encoded successfully.")
    

    def split_data(self, target_y_column, target_x_columns = None, test_size = 0.2):
        if target_x_columns == None:
            X = self.data.drop(columns=[target_y_column])
            
            
        else:
            cols_to_drop = []
            for i in list(self.data):
                print(i)
                if i in target_x_columns:
                    inx = 1
                else:
                    cols_to_drop.append(i)
                   
            X = self.data.drop(columns=cols_to_drop)
        y = self.data[target_y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    
    
    def ordered_split_data(self, target_y_column, target_x_columns = None, test_value = -5, cull = False, n = 30):
        

        
        od = self.data
        
        
        

        
        if target_x_columns == None:
            od = od.sort_values(target_y_column)
            
            
            if cull == True:
                start = random.randint(0, len(od.index - n-1))
        
            
                df = od.iloc[start:start+n+1]
            

                od = df
            
            
            else: 
                n = n
            
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
                start = random.randint(0, len(od.index - n-1))
        
            
                df = od.iloc[start:start+n+1]
            

                od = df
            
            
            else: 
                n = n
            
            
            X = od.drop(columns=cols_to_drop)
        y = od[target_y_column]
        X_train = X.iloc[0:test_value]
        y_train = y.iloc[0:test_value]
        
        X_test = X.iloc[test_value:]
        y_test = y.iloc[test_value:]
        
        
        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
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
    def select_and_drop(self, n = 30):
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
    
    def orderd_and_drop(self, n = 30):
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
        
        df = self.data.iloc[start:start+n+1]
        

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

    '''
    AT: - We should create functions to filter the dataset (maybe based on country or timeframe?)
    AT: - And we should prompt the player to select which filter they want to apply
    AT: starting the functions as below...

    def get_unique_values(self):
        unique_values = df['column_name'].unique()
        return unique_values
    
    def filter_data(self):
        filtered_df = df[df['column_name'].isin(unique_values)]
        return filtered_df
    '''




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


    def evaluate(self, X, y):
        """
        Evaluate the model's performance.
        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
        Returns:
            tuple: (Mean squared error, R-squared score)
        """
        # LM in this case is y_pred, 'linear model'
        LM_pred = self.predict(X)
        mse = mean_squared_error(y, LM_pred)
        r2 = r2_score(y, LM_pred)
        mape = mean_absolute_percentage_error(y, LM_pred)
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

    def polynomial_model(degree):
        """Creates a polynomial regression model of a given degree."""
        return make_pipeline(PolynomialFeatures(degree), LinearRegression())
        '''
        Example usage polynomial_model:
        model = create_polynomial_model(2)
        model.fit(X, y)

        # Predict values
        y_pred = model.predict(X)
        print(y_pred)
        '''


# How to Use the ModelManager Class ----------------------
# Create an instance of the class
'''
model = ModelManager()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)

# Get coefficients and intercept
coefficients = model.get_coefficients()
intercept = model.get_intercept()
'''


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




# Ready for ResultVisualizer Class !!! --------------------------------------------------------------------------
'''
def auto_reg_lin(x, y, X_test):
    x = np.array(x)
    y = np.array(y)
    X_test = np.array(X_test)
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
'''

'''
# ----- RUN THE BACKEND PROGRAM !!! -----------------------------------------------------------------------------
# Call the DataLoader Class -------------------------------------------------------------------------------------
print('Backend code testing/debugging beginning')
print("\n")
print("Beginning program execution \n")
print("Calling the DataLoader Class \n")
# Call the methods from the DataLoader class
data_loader = DataLoader(sales_data)
data_loader.preprocess_data()
data_loader.handle_missing_values()

# Split the data (assuming 'SALES' is the target column)
X_train, X_test, y_train, y_test = data_loader.split_data(target_y_column='SALES', target_x_columns=['ORDERDATE'])
print('X_train data: \n', X_train)


# Output first few rows of training data for verification
print("First few rows of training data: \n", X_train.head())

# Display summary statistics
print("Here are the summary statistics of the Sales Dataframe: \n")
summ_stats_df = DataLoader(sales_data_df)
summ_stats_df.preprocess_data()
summ_stats_df.handle_missing_values()
summary_stats = summ_stats_df.get_summary_stats()
print("\n")
print("Summary Statistics on the DataFrame:")
print(summary_stats)

# Dataset is preprocessed and cleaned ready for testing and predictions !!!


# Call the ModelManager Class -----------------------------------------------------------------------------------
# Call the methods from the ModelManager class
print("\nCalling the ModelManager Class to Fit the LinearRegression Model")
regr = ModelManager() #create instance of the ModelManager class
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse, r2, mape = regr.evaluate(X_test, y_test)
coefficients = regr.get_coefficients()
intercept = regr.get_intercept()
weights = regr.return_weights()
print('Weights: ', weights)

print("Here are the results of the fitted linear model: ")
print("MSE: ", mse)
print("r2: ", r2)
print("MAPE: ", mape)
print("\n")

print("Backend Program Executed Successfully \n \n \n")
print("---------------------------------------------------- \n")


# Convert to arrays for Flask
"""
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
"""
'''
'''
data_loader = DataLoader(sales_data)
data_loader.select_and_drop()
print('Here is the row count of the dataframe: ',data_loader.row_count())
data_loader.preprocess_data()
data_loader.handle_missing_values()
print('Here is the row count of the processed dataframe: ',data_loader.row_count())
'''

# END of BACKEND PROGRAM ----------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# START FRONT END PROGRAM AND GUI -------------------------------------------------------------------------------


# Eventually reshuffle all the code around so this makes sense with the backend program -------------------------
# START GAMEPLAY LOOP -------------------------------------------------------------------------------------------
'''print('Welcome to Xtrapolate: Gamified Data Science!')
print("In this game, you will attempt to guess values based on some graphed data.")
print("Are you ready to play? \n")

# Initialize the game
game = GameManager(topics, difficulty, 0, 1)
print('Calling the Game:\n')


# player selects difficulty
game.display_difficulty()
selected_difficulty = game.get_difficulty_selection()
print(f"\nYou selected: {selected_difficulty}")

# player selects topic
game.display_topics()
selected_topic = game.get_topic_selection()
print(f"\nYou selected: {selected_topic}")

# player filters the data
# insert code here once filter functions working on dataset


# Display the summary statistics of the data
print("\n")
print("Here are the summary statistics of the Sales Data: \n")
summ_stats_df = DataLoader(sales_data_df)
summ_stats_df.select_and_drop()
summ_stats_df.preprocess_data()
summ_stats_df.handle_missing_values()
summary_stats = summ_stats_df.get_summary_stats()
print(summary_stats)

'''
# Begin Charting / Load Flask & Bokeh ----------------------------------------------------------------------
'''
print("\n")
print('Here is a chart of the data:')
# Display the chart here ----------------


# Debugging for Flask/Bokeh
#print("X test data: ", X_test[0])

# Run the Application Start Bokeh Charts -------------------------------------------------------------------
Plot_Title = 'Dummy Data'
'''
def bridge(data_set, cull = False):
    
    
    # this is essentialy hard coded for now but I think making different paths for each dataset based on the collums we actually use makes sense, can be altered later.
    if data_set == "Sales":
        
        data_loader = DataLoader(sales_data)
        
        # add filter productline !!!
        # inser filter for productline here
        data_loader.preprocess_data()
        data_loader.handle_missing_values()
        
        #data_loader.orderd_and_drop()
        
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


@app.route('/')
def start_page():
    #return render_template('start_page.html')
    return f'''
    <html lang="en">
    <body>      
    <h1>Welcome to Xtrapolate!</h1>
    <h2>a data science game</h2>
    <h3>Created by: Thomas Taylor, Jomaica Lei, Andy Turner</h3>
    <form action="/guess" method = "POST">
    <p><input type = "submit" value = "Start game" /></p>
    </form>
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
        
        X_train, X_test, y_train, y_test,  Plot_Title = bridge("Sales", True)
        
        test1 = [1, 2, 3]
        test2 = [4, 5, 6]
        
        p = figure(height=350, x_axis_type='datetime', sizing_mode="stretch_width")
        p.add_tools(HoverTool())
        # Defining Plot to be a Scatter Plot
        p.circle( 	[i for i in X_train],
    		[j for j in y_train],
            size=20,
            color="blue",
            alpha=0.5
        )
        
        # Get Chart Components
        script, div = components(p)
    
    
        # Return the components to the HTML template
        return f'''
    	<html lang="en">
    		<head>
    			<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.3.min.js"></script>
    			<title>Bokeh Charts</title>
    		</head>
    		<body>
    			<h1> Graph of {Plot_Title} </h1>
    			{ div }
    			{ script }
                
                
                <h1> Submit a prediction for Y at the following X values </h1>
                
                <form action="/display" method = "POST">
        <p> {str(X_test[0]):.10} <input type = "number" step = "any" name = "g1" required /></p>
        <p> {str(X_test[1]):.10} <input type = "number" step = "any" name = "g2"  required /></p>
        <p> {str(X_test[2]):.10} <input type = "number" step = "any" name = "g3" required /></p>
        <p> {str(X_test[3]):.10} <input type = "number" step = "any" name = "g4" required /></p>
        <p> {str(X_test[4]):.10} <input type = "number" step = "any" name = "g5" required /></p>
        <p><input type = "submit" value = "Submit" /></p>
        </form>
                
    		</body>
    	</html>
    	'''



@app.route('/display/', methods = ['POST', 'GET'])
def display():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        
        X_train, X_test, y_train, y_test,  Plot_Title = bridge("Sales")
        
        
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
        

        s = ScoreManager()
        
        user_score = s.scoring(user_guesses, y_test, weights, ybar)
        
        

        regr = ModelManager()
        
        #ntercept, coefficients, ML_pred = regr.auto_reg_lin(X_train, y_train, X_test)
        
        
        regr.fit(train_days.reshape(-1, 1), y_train)
        
        gamecoefficients = regr.get_coefficients()
        intercepts = regr.get_intercept()
        ML_pred = regr.predict(test_days.reshape(-1, 1))
        ML_Score = s.scoring(ML_pred, y_test, weights, ybar)
        p = figure(height=350, sizing_mode="stretch_width")
        p.add_tools(HoverTool())

        # Defining Plot to be a Scatter Plot
        p.circle(
            [i for i in X_train],
            [j for j in y_train],
            size=20,
            color="blue",
            alpha=0.8,
            legend_label = "Actual value"
        )
        
        p.circle(
            [i for i in X_test],
            [j for j in y_test],
            size=20,
            color="blue",
            alpha=0.8
        )
        
        p.circle(
            [i for i in X_test],
            [j for j in user_guesses],
            size=10,
            color="orange",
            alpha=0.9,
            legend_label = "User Predicted value"
        )
        
        p.circle(
            [i for i in X_test],
            [j for j in ML_pred],
            size=5,
            color="red",
            alpha=0.8,
            legend_label = "ML Predicted value"
        )
        
        p.legend.location = 'top_left'

        # Get Chart Components
        script, div = components(p)
        if user_score >= ML_Score: 
            return  f'''
        <html lang="en">
            <head>
                <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.3.min.js"></script>
                <title>Bokeh Charts 2</title>
            </head>
            <body>
                <h1> After </h1>
                { div }
                { script }
                <h2> User Score was: {user_score}, ML Score Was {ML_Score } <h2>
                <h2> User Wins </h2>
                <form action="/guess" method = "POST">
        <p><input type = "submit" value = "Continue Playing" /></p>
        </form>
                <form action="/" method = "POST">
        <p><input type = "submit" value = "Return to Homepage" /></p>
        </form>
            </body>
        </html>
        '''
        else:
            return  f'''
        <html lang="en">
            <head>
                <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.3.min.js"></script>
                <title>Bokeh Charts 2</title>
            </head>
            <body>
                <h1> After </h1>
                { div }
                { script }
            <h2> User Score was: {user_score}, ML Score Was {ML_Score } <h2>
                <h2> ML Wins </h2>    
        <form action="/guess" method = "POST">
<p><input type = "submit" value = "Continue Playing" /></p>
    </form>
        <form action="/" method = "POST">
<p><input type = "submit" value = "Return to Homepage" /></p>
</form>
            </body>
        </html>
        '''
        
        

    
X_train, X_test, y_train, y_test,  Plot_Title = bridge("Sales") 
#app.run(debug=False)

if __name__ == '__main__':
	# Run the application on the local development server
	app.run(debug=True)
