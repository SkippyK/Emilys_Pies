#!/usr/bin/env python
# coding: utf-8


#Emily_dashboard_project for deployment on pythonanywhere.com version 2
#Lovingly created by Skippy Keiter 01/27/25 updated 02/03/25

#Imports
from dash import Dash, dcc, html, no_update, dash_table, State, callback_context
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

#styling components
import dash_bootstrap_components as dbc

# dependent librarires
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import base64
import io

# Timing and logging for performance eval
import time
import logging

logging.basicConfig(level = logging.INFO)

#Note: Need to install xlrd and openpyxl to read both .xls and .xlsx files

app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

app.layout = html.Div(
             [
                dcc.Tabs(
                    id = 'TABS',
                    value = 'Tab1',    # this sets Tab1, the upload tab, as the default active tab for uploading data file
                    children = [
                        dcc.Tab(
                            label = 'Pie Charts',
                            value = 'Tab2',
                            id = 'Tab2',
                            disabled = True,
                            style={
                                'fontSize': '16px',          # Font size
                                'fontFamily': 'Arial',      # Font style
                                'padding': '10px',          # Padding inside the tab
                                'backgroundColor': '#f2f2f2',  # Background color
                            },
                            selected_style={
                                'fontSize': '18px',         # Larger font for selected tab
                                'fontWeight': 'bold',       # Bold font for emphasis
                                'padding': '10px',
                                'backgroundColor': '#d1e7dd',  # Highlighted background color
                                'color': '#0c4128',         # Text color
                            },
                            #Note: all content for Tab1 goes in Tab1 children below
                            children = [dbc.Container(
                                [
                                    html.H1(children = ['Choose Kiddo'], id = 'Header'),
                                    html.Button('Choose new kiddo', id='reset-button', n_clicks=0, style = {'display' : 'none'}),
                                    #Note: check duplicates should on upload of CSV file
                                    dcc.Markdown(
                                        id = 'check_duplicates'
                                    ),
                                    dcc.Dropdown(
                                        id = 'full-name-dropdown',
                                        placeholder = 'Select kiddo'
                                    ),
                                    html.Div(id = 'full-name-display',
                                            children = []),
                                    html.Div(id = 'PieFigure',
                                             children = []),
                                
                            ],    #end of container children
                            fluid = True,
                            ),    #end of container arguments
                            ]    #end of Tab2 children
                        ),    # end of Tab2 arguments
                        #upload Tab
                        dcc.Tab(
                            label = 'Upload Data',
                            value = 'Tab1',
                            id = 'Tab1',
                            style={
                                'fontSize': '16px',          # Font size
                                'fontFamily': 'Arial',      # Font style
                                'padding': '10px',          # Padding inside the tab
                                'backgroundColor': '#f2f2f2',  # Background color
                            },
                            selected_style={
                                'fontSize': '18px',         # Larger font for selected tab
                                'fontWeight': 'bold',       # Bold font for emphasis
                                'padding': '10px',
                                'backgroundColor': '#d1e7dd',  # Highlighted background color
                                'color': '#0c4128',         # Text color
                            },
                            #Tab1 children
                            children = [dbc.Container([
                                            html.H1("To get started upload your spreadsheet"),
                                            html.H3("(file format: .csv, .xls, or .xlsx)"),
                                            dcc.Upload(
                                                id='upload-data',
                                                children=html.Div(['Drag and Drop or ', html.A('Select a CSV or Excel File')]),
                                                style={
                                                    'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px'
                                                },
                                                multiple=False  # Single file at a time
                                            ),
                                            dcc.Store(id = 'df-storage'),
                                            html.Div(id='output-data-upload'),
                                        ]    #end of children container for tab3
                                                     )    #end of container for tab3
                                       ]    #end of tab1 children
                        )    #end of tab1 arg
                    ]    #end of Tabs children
                )    # end of Tabs Arguments
            ]    #end of main html.Div children
        )    #end of html.Div arguments

#Tab1 callback functions for uploading the spreadsheet file
                                        
def parse_spreadsheet(contents, filename):
    """Parse the uploaded file into a Pandas DataFrame with encoding handling."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.endswith(('.csv')):
        # Try to read the CSV with default encoding (UTF-8) first
        try:
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except UnicodeDecodeError:
            # If UTF-8 fails, try a fallback encoding
            try:
                return pd.read_csv(io.StringIO(decoded.decode('ISO-8859-1')))
            except Exception as e:
                return f"Error processing CSV file: {e}"
                
    elif filename.endswith('.xls'):
        try:
            return pd.read_excel(io.BytesIO(decoded), engine='xlrd')  # Use xlrd for .xls files
        except Exception as e:
            return f"Error processing Excel file (.xls): {e}"

    elif filename.endswith('.xlsx'):
        try:
            return pd.read_excel(io.BytesIO(decoded), engine='openpyxl')  # Use openpyxl for .xlsx files
        except Exception as e:
            return f"Error processing Excel file (.xlsx): {e}"
            
    else:
        return f"Error in parsing spreadsheet: Check file integrity by opening the file in excel or google sheets: {filename}"

# function to update the Outputs
def update_callback(change_tups):
    ''' input -->list of tuples--each tuple (position of output, change to be made).
        For handling long lists of outputs in a return statement
        eg. position 4 is Output(#5) for the options of the last name dropdown and change it to [] (empty options)
        would be (4, []) the remaining outputs would be no_update'''
    
    # Define default return values (all no_update)
    return_values = [no_update] * len(callback_context.outputs_list)  # Number of outputs in your callback
    if change_tups:
        for change_tup in change_tups:
            # Modify only necessary values
            return_values[change_tup[0]] = change_tup[1]

    # Convert back to tuple and return
    return tuple(return_values)
 
#Tab1 upload functions

@app.callback(
    Output('output-data-upload', 'children'),
    Output('df-storage', 'data'),
    Output('Tab2','disabled'),
    # Output('last-name-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    '''Display the DataFrame as a preview in the dashboard 
       Check for duplicate names and resolve an ambiguity, error message for unresolvable ambiguity
       convert df to json for storage in dcc.Store for future use
       Note: can modify the df (add/subtract...) here before storage into Store
       unhide the other tab'''
    #Performance eval
    start_time = time.time()
    
    if contents is not None:
        if filename.endswith(('.csv','.xls', '.xlsx')):
            
            # read file with parse_spreadsheet function
            df = parse_spreadsheet(contents, filename)
            
            if isinstance(df, pd.DataFrame):
                # modify the df (Note: pandas auto changes dtype to date format for columns labeled date)
#########Future: can change all date columns to date dtype format for slider that regresses in time
                df.rename(columns = {"Last Name": "Last_Name", "First Name": "First_Name", "Date": "Date.0"}, inplace = True)
                
                #check for duplicate names
                #Modify the dataset to resolve ambiguity of duplicate names by adding teacher to first name
                df['Is_Duplicate'] = df.duplicated(subset=['First_Name', 'Last_Name'], keep=False)    #Note: keep = False marks all duplicates as True in the is_duplicate column, not just the second as true
                # overwrite First_name column with unique identifiers
                df['First_Name'] = df.apply(
                                        lambda row: f"{row['First_Name']}({row['Teacher']})" if row['Is_Duplicate'] else row['First_Name'],
                                        axis=1)
                # After resolving duplicates by modifying first name, check again for duplicates
                # If still dups-- get names of dups and present error message with dups names for fixing
                if df.duplicated(subset=['First_Name', 'Last_Name'], keep=False).any():
                    # preview the names in the duplicates-df to tell the user which students need fixing
                    dups = df[df.duplicated(subset = ['First_Name', 'Last_Name', 'Teacher'], keep=False)]
                    dups = dups[['First_Name', 'Last_Name', 'Teacher']]
                    return html.Div([
                        html.H3(f'Unresolvable duplicates found. Please check spreadsheet for the following duplicates:'),
                        html.Pre(dups.to_string(index = False)),
                        html.H6("Refresh the page to upload a new file")
                    ]), no_update, no_update

                # clean the data to make sure that all empty cells are NaN
                df = df.applymap(lambda x: pd.NA if isinstance(x, str) and x.strip() == '' else x)
                
                #check for categorical performance column
                # if the column name doesn't exist then check if it might be named something else
                if not "Performance Band RP2.1" in df.columns:
                    #check the df columns for the following values
                    cat_values = {"Above", "Benchmark", "Below", "Well Below"}
                    
                    # Function to detect the categorical column
                    def find_categorical_column(df, values):
                        # print('running function to find cat columns')
                        for col in df.columns:
                            if df[col].astype(str).isin(values).any():  # Convert to string to handle mixed types
                                # print('found column with categorical values: ', col)
                                return col  # Return the first matching column
                        return None  # Return None if no matching column is found
                    
                    # Check for the categorical column with the above function
                    cat_col = find_categorical_column(df, cat_values)
                    # print('the function returned: ', cat_col)
                    if cat_col:
                        #the column exists rename it "Performance Band RP2.1"
                        # print('checking what the function returned--returned a col name that is not perform band rp2.1: ', cat_col)
                        df.rename(columns = {cat_col: "Performance Band RP2.1"}, inplace = True)
                        # print('the renamed columns: ', df.columns)
                    else:
                        # the column doesn't exist--create a replacement column
#####Future: set the column values according to the cutoffs--currently it is set to 'No_Value', 
                        # "No_Value" can be set in the dictionary as white background and white heart
                        # print('cat_col should be None here and so creating the column')
                        df["Performance Band RP2.1"] = 'No_Value'
                        # print('now that column has been created, checking that it is accessible: ', df["Performance Band RP2.1"])
                
                #make a new column for the emojis to reference later
                performance_emojis = {'Above' :'💙', 'Benchmark' : '💚' , 'Below' : '💛' ,'Well Below' : '❤️', 'No_Value': '🤍'}   # emojis need to be treated as strings
                df['Emojis'] = df['Performance Band RP2.1'].map(performance_emojis)
                
                #sort the df by performance
                performance_order = ['Well Below', 'Below', 'Benchmark', 'Above', 'No_Value']
                df['performance_category'] = pd.Categorical(df["Performance Band RP2.1"], categories = performance_order, ordered = True)
                df.sort_values(by = ['performance_category', 'First_Name'], inplace = True)
                
                #Performance eval
                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"Upload Callback execution time: {elapsed_time:.3f} seconds")

#########Future: Preview a df that has the name of the subtests and the max-score of each in case these values need to be adjusted

                return html.Div([
                    html.H5(f"Uploaded File: {filename}"),
                    html.H6("Data Preview:"),
                    html.Pre(df.head().to_string(index=False),),
                    html.H6('Refresh to upload new file')
                ]), df.to_json(date_format='iso', orient='split'), False
                
            else:
                # Return an error message if data is not a dataframe after parsing, it will be an error message as a string
                return html.Div([html.H5(df)]), no_update, no_update
        else:
            # Return an error message if filename does not end in csv, xls, xlsx
            return html.Div([html.H5(f"Error: {filename} not in proper format, need .csv, .xls, or .xlsx")]), no_update, no_update
    else: 
        raise PreventUpdate

#Tab2-PieChart Functions
    
@app.callback(
    [
    Output('full-name-dropdown', 'style'),      #0
    Output('full-name-dropdown', 'options'),    #1
    Output('full-name-dropdown', 'value'),      #2  
    Output('PieFigure', 'children'),            #3
    Output('reset-button', 'n_clicks'),        #4
    Output('reset-button', 'style'),           #5
    Output('Header', 'children')               #6
    ],
    [Input('full-name-dropdown', 'value'),
     # Input('first-name-dropdown', 'value'),
     Input('reset-button', 'n_clicks'),
     Input('df-storage', 'data')
    ]
)

def PieFig(full_name, n_clicks, json_data):
    '''When full name is chosen make pie fig
    --for items that do not need updating use no_update
    '''    

    if not json_data and not full_name:
        raise PreventUpdate
    
#### Pie chart function
    def PieCharts(student_data):
        # get all the date columns and collect the names of the tests
        # list all the date columns
        date_columns = [col for col in student_data.columns if 'date' in str(col).lower()]
        # list all the subtest columns
        subtest_columns = []
        for col in date_columns:
            date_col_index = student_data.columns.get_loc(col)
            subtest_col_index = date_col_index + 1
            subtest_columns.append(student_data.columns[subtest_col_index])

#######Future: check that columns with subtest values are numbers


        #maximum score for each subtest
        max_scores = [21,5,10,5,5,5,5,5,5,5,5,5,5,5]
        #max score dictionary
        max_dict = {k:v for k,v in zip(subtest_columns, max_scores)}
        
        #Figure layout parameters
        n_test = len(subtest_columns)
        columns = 7
        rows = ((n_test + 1)//columns)
        
        # get the dates that each subtest was given from the filtered data and add this info to subplot_titles
        subplot_titles = []
        date_columns = [student_data[col].item() for col in student_data.columns if 'date' in str(col).lower()]
        for item in zip(date_columns, subtest_columns):
            if pd.isna(item[0]):
                subplot_title = f'{item[1]}'
            else:
                subplot_title = f'{item[1]}<br>{item[0]}'
            subplot_titles.append(subplot_title)
        
        #set up empty figure based on layout parameters
        fig = make_subplots(rows = rows,
                            cols = columns,
                            subplot_titles = subplot_titles,
                            specs=[[{'type': 'domain'}] * columns for _ in range(rows)]
                           )
        fig.update_annotations(font=dict(size=10))

        
            #color mapping for background of figure
        # Search for a column name containing performance eval(no a priori knowledge of column name)
        target_values = {'Above', 'Benchmark', 'Below', 'Well Below'}
        performance_col = [col for col in student_data.columns if student_data[col].astype(str).isin(target_values).any()]
        
        # correlation map of background color to performance benchmark
        colormap = {'Above':'lightskyblue', 'Benchmark':'#90EE90','Below':'lemonchiffon','Well Below':'lightpink', 'No_Value': 'white'}
        
        # if a categorical performance col is found then get category and set background color using colormap
        if performance_col:
            try:
                background_color = colormap[student_data[performance_col[0]].item()]
                #get overall score for fig title
                col_before = student_data.columns[(student_data.columns.get_loc(performance_col[0]) - 1)]
                score = student_data[col_before].item()
            except:
                background_color = 'white'
                score = ' '
        else:
            background_color = 'white'
            score = ' '
        # added full_name to figure title-- so that if the figure is downloaded it retains which student it is.  
        full_name = f"{student_data['First_Name'].item()} {student_data['Last_Name'].item()}"
        title_text = f"{full_name}, Performance Band RP2 Overall:  {score}"
        fig.update_layout(height=325 * rows, 
                          width=1300, 
                          title_text=title_text,
                          paper_bgcolor=background_color, 
                          plot_bgcolor=background_color)
        #end of color mapping         
        ### pie chart loop--loop through
        
        for i, col in enumerate(subtest_columns):
            
            correct = student_data[col].iloc[0]
            
            if pd.isna(correct):
                temp_df = {'Category': ['Not Taken', 'WIP'], 'Count': [0,5]}  # Dummy data
                color_map = {'Not Taken': 'lightgrey', 'WIP': 'grey'}
            else:
                #calculate incorrect using the subtest_column name and the max score dictionary
                incorrect = max_dict[col] - correct
                temp_df = {'Category': ['Yeah', 'WIP'], 'Count': [correct, incorrect]}
                color_map = {'WIP' : 'red', 'Yeah' : 'green'}
                
            pie_fig = px.pie(temp_df, values = 'Count',
                             names = 'Category',
                             color = 'Category',
                             color_discrete_map = color_map,
                             category_orders = {'Category': ['Yeah', 'WIP']}    #only works with python 3.10 and plotly 5.22
                      )
            # collect the pie chart data into the figure and assign proper location
            fig.add_trace(
                pie_fig.data[0],
                row=(i // columns) + 1,
                col=(i % columns) + 1
            )
        return fig
### End of PieChart Function
        
    #logging time performance
    start_time = time.time()

    if json_data:
        df = pd.read_json(io.StringIO(json_data), orient = 'split')
         # Check the reset-button after json present and before other parameters
            #reset-button click sets n_clicks to 1 this will reset the state parameters to json_data = True, full_name = None
        if n_clicks > 0:
            n_clicks = 0    # reset the counter so as not to trigger again until clicked
            #Note: two spaces are needed in the value for full name options to split correctly later
            fullname_options = [{'label': f'{first} {last} {heart}', 'value': f'{first}  {last}'} for first, last, heart in zip(df['First_Name'], df['Last_Name'], df['Emojis'])]
            changes = [(0,{'display': 'block'}),    #unhide full name dropdown
                       (1, fullname_options),
                       (2, None),
                       (3, []),
                       (4, n_clicks),
                       (5, {'display': 'none'}),    #hide reset button
                       (6, ['Choose Kiddo'])
                      ]
            return update_callback(changes)

        if not full_name:
            #set the full name options
            fullname_options = [{'label': f'{first} {last} {heart}', 'value': f'{first}  {last}'} for first, last, heart in zip(df['First_Name'], df['Last_Name'], df['Emojis'])]
            changes = [(1, fullname_options)]
            return update_callback(changes)
        else:
            #Note: fullname-value has two spaces between first and last name to allow for proper split here
            first_name, last_name = full_name.split('  ')
            student_data = df.query("First_Name == @first_name & Last_Name == @last_name")
        
            fig = PieCharts(student_data)
        
            #performance eval
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"PieFig Callback execution time: {elapsed_time:.3f} seconds")
        
            changes = [(0, {'display': 'none'}),
                       (3, dcc.Graph(id = 'piefigure', figure = fig)),
                       (5,{'display' : 'block'}),
                       (6, [full_name])
                      ]
            return update_callback(changes)
