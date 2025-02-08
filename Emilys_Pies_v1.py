#!/usr/bin/env python3.8
# coding: utf-8

#Emily_dashboard_project for deployment on pythonanywhere.com
#Lovingly created by Skippy Keiter 01/27/25

#Imports
from dash import Dash, dcc, html, no_update, dash_table, State
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
                                    #Note: check duplicates should on upload of CSV file
                                    dcc.Markdown(
                                        id = 'check_duplicates'
                                    ),
                                    dcc.Dropdown(
                                        id = 'last-name-dropdown',
                                        # options = [],
                                        # options = [{'label': last_name, 'value': last_name }for last_name in df['Last Name'].unique()],
                                        placeholder = 'Select a Last Name',
                                    ),
                                    dcc.Dropdown(
                                        id = 'first-name-dropdown',
                                        placeholder = 'Select a First Name',
                                        style = {'display' : 'none'}
                                    ),
                                    html.Div(id = 'full-name-display',
                                            children = []),
                                    html.Div(id = 'PieFigure',
                                             children = []),
                                    html.Button('Choose new kiddo', id='reset-button', n_clicks=0, style = {'display' : 'none'}),
                            




                                
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
    elif filename.endswith(('.xls', '.xlsx')):
        try:
            return pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            return f"Error processing Excel file:  {e}"
            
    else:
        return f"Error: Unsupported file type: {filename}"

#Tab1 upload functions

@app.callback(
    Output('output-data-upload', 'children'),
    Output('df-storage', 'data'),
    Output('Tab2','disabled'),
    Output('last-name-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    '''Display the DataFrame as a preview in the dashboard 
       Check for duplicate names and resolve an ambiguity, error message for unresolvable ambiguity
       convert df to json for storage in dcc.Store for future use
       Note: can modify the df (add/subtract...) here before storage into Store
       unhide the other tab'''

    if contents is not None:
        if filename.endswith(('.csv','.xls', '.xlsx')):
            # read file with parse_spreadsheet function
            df = parse_spreadsheet(contents, filename)
            if isinstance(df, pd.DataFrame):
                # modify the df (Note: pandas auto changes dtype to date format for columns labeled date)
                #######Future: can change all date columns to date dtype format for slider that regresses in time
                df.rename(columns = {"Last Name": "Last_Name", "First Name": "First_Name", "Date": "Date.0"}, inplace = True)
                #check for duplicate names
                #Modify the dataset to resolve ambiguity of duplicate names by adding teacher to first name
                df['Is_Duplicate'] = df.duplicated(subset=['First_Name', 'Last_Name'], keep=False)    #Note: keep = False marks all duplicates as True in the is_duplicate column, not just the second as true
                # overwrite First_name column with unique identifiers
                df['First_Name'] = df.apply(
                                        lambda row: f"{row['First_Name']} ({row['Teacher']})" if row['Is_Duplicate'] else row['First_Name'],
                                        axis=1)
                # If names not unique send error message to check spreadsheet for duplicate students
#####Future: In error message put in the name of the duplicate, working this out below
                if df.duplicated(subset=['First_Name', 'Last_Name'], keep=False).any():
                    # preview the names in the duplicates-df to tell the user which students need fixing
                    dups = df[df.duplicated(subset = ['First_Name', 'Last_Name', 'Teacher'], keep=False)]
                    dups = dups[['First_Name', 'Last_Name', 'Teacher']]
#########Future: Preview a df that has the name of the subtests and the max-score of each in case these values need to be adjusted
                    return html.Div([
                        html.H3(f'Unresolvable duplicates found. Please check spreadsheet for the following duplicates:'),
                        html.Pre(dups.to_string(index = False)),
                        html.H6("Refresh the page to upload a new file")
                    ]), no_update, no_update, no_update
                # if all duplicates have been handled then continue on--no else statement required
                # options for last name dropdown
                options = [last_name for last_name in df["Last_Name"].unique()]
                return html.Div([
                    html.H5(f"Uploaded File: {filename}"),
                    html.H6("Data Preview:"),
                    html.Pre(df.head().to_string(index=False),),
                    html.H6('Refresh to upload new file')
                ]), df.to_json(date_format='iso', orient='split'), False, options
            else:
                # Return an error message if data is not a dataframe after parsing
                return html.Div([html.H5(f"Unknown Error: {filename} data could not be read, check format")]), no_update, no_update, no_update
        else:
            # Return an error message if filename does not end in csv, xls, xlsx
            return html.Div([html.H5(f"Error: {filename} not in proper format, need .csv, .xls, or .xlsx")]), no_update, no_update, no_update
    else: 
        raise PreventUpdate

#Tab2-PieChart Functions
    
@app.callback(
    [Output('first-name-dropdown', 'style'),
    Output('first-name-dropdown', 'options'), 
     Output('first-name-dropdown', 'value'),        
    Output('last-name-dropdown', 'style'), 
     Output('last-name-dropdown', 'value'),
    Output('PieFigure', 'children'),
     Output('reset-button', 'n_clicks'),
     Output('reset-button', 'style'),
     Output('Header', 'children')
    ],
    [Input('last-name-dropdown', 'value'),
     Input('first-name-dropdown', 'value'),
     Input('reset-button', 'n_clicks'),
     Input('df-storage', 'data')
    ]
)

def PieFig(last_name, first_name, n_clicks, json_data):
    '''Check for both last name and first name, if neither-prevent update, if lastname only 
    update the first name dropdown with possible choices, if both then proceed with generating
    a figure of pie charts for each test in the dataframe
    --for items that do not need updating use no_update
    '''

    # Pie chart function
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
        date_columns = [str(student_data[col].item()) for col in student_data.columns if 'date' in str(col).lower()]
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
        # Search for a column name containing performance eval
        target_values = {'Above', 'Benchmark', 'Below', 'Well Below'}
        performance_col = [col for col in student_data.columns if student_data[col].astype(str).isin(target_values).any()]
        
        # correlation map of background color to performance benchmark
        colormap = {'Above':'lightskyblue', 'Benchmark':'#ADFF2F','Below':'lemonchiffon','Well Below':'lightpink'}
        
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
        fig.update_layout(height=325 * rows, 
                          width=1300, 
                          title_text=f"Performance Band RP2 Overall:  {score}",
                          paper_bgcolor=background_color, 
                          plot_bgcolor=background_color)
#end of color mapping         
### pie chart loop
        
        for i, col in enumerate(subtest_columns):
            
            correct = student_data[col].iloc[0]
            
            if pd.isna(correct):
                temp_df = {'Category': ['Not Taken', 'WIP'], 'Count': [0,5]}  # Dummy data
                color_map = {'Not Taken': 'lightgrey', 'WIP': 'grey'}
                # title = f"{col}<br>(No Data)"
            else:
                #calculate incorrect using the subtest_column name and the max score dictionary
                incorrect = max_dict[col] - correct
                temp_df = {'Category': ['Yeah', 'WIP'], 'Count': [correct, incorrect]}
                color_map = {'WIP' : 'red', 'Yeah' : 'green'}
                # title = col
                
            pie_fig = px.pie(temp_df, values = 'Count',
                       names = 'Category',
                       color = 'Category',
                       color_discrete_map = color_map,
                       category_orders = {'Category': ['Yeah', 'WIP']}
                      )
            # collect the pie chart data into the figure and assign proper location
            fig.add_trace(
                pie_fig.data[0],
                row=(i // columns) + 1,
                col=(i % columns) + 1
            )
        return fig
    ##End of PieChart Function
  
    if json_data:
        df = pd.read_json(io.StringIO(json_data), orient = 'split')
    
    else:
        raise PreventUpdate

    # reset button--return to start state to choose another kiddo
    if n_clicks > 0:
        n_clicks = 0
        #return all output values to the start/default state
        return {'display': 'none'}, [], None, {'display': 'block'}, None, [], n_clicks, {'display': 'none'}, ['Choose Kiddo']
    
    if not last_name:
        raise PreventUpdate
        
    elif last_name and not first_name:
        data = df.query("Last_Name == @last_name")        
        options = [{'label': name, 'value': name} for name in data['First_Name']]
        return  {'display' : 'block'}, options, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    elif last_name and first_name:
        # filter data
        student_data = df.query("First_Name == @first_name & Last_Name == @last_name")
        full_name = f'{first_name}  {last_name}'        

        #run the pie chart function
        fig = PieCharts(student_data)

        return ({'display': 'none'},
                no_update, 
                no_update,
                {'display': 'none'},
                no_update,
                dcc.Graph(id = 'piefigure', figure = fig),
                no_update,
                {'display' : 'block'},
                [full_name])

server = app.server