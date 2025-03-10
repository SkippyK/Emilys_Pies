#!/usr/bin/env python
# coding: utf-8

#Emily_dashboard_project for deployment on pythonanywhere.com version 3
#Lovingly created by Skippy Keiter 01/27/25 

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
from functools import reduce

# Timing and logging for performance eval
import time
import logging

logging.basicConfig(level=logging.INFO, force=True)  # 'force' resets previous handlers
logger = logging.getLogger(__name__)


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
                                    html.Div(id = 'Error_message'),
                                    html.H1(children = ['Choose Kiddo'], id = 'Header'),
                                    html.Button('Choose new kiddo', id='reset-button', n_clicks=0, style = {'display' : 'none'}),
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
                                            html.H1("To get started upload your spreadsheet in excel format (.xls or .xlsx)", id = 'upload-H1'),
                                            html.H4("To upload a different file, refresh the page -->  🔄 ", id = 'upload-H3'),
                                            html.Br(),
                                            dcc.Upload(
                                                id='upload-data',
                                                children=html.Div(['Drag and Drop or ', html.A('Click here to select a file')]),
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
                                            dcc.Store(id = 'max-dict'),
                                            html.Div(id = 'sheet-selector-div',                                            
                                                     children = [
                                                                 dbc.Row(
                                                                         children = [dbc.Col(html.H3('Pick a sheet, any sheet!  ',
                                                                                            id = 'pick-a-sheet',                                                                                            
                                                                                                    ), 
                                                                                             width = 3
                                                                                            ),  
                                                                                    dbc.Col(dcc.Dropdown(id = 'sheet-selector'))
                                                                                    ],
                                                                         style = {
                                                                                    'width': '100%',
                                                                                    'height': '80px',
                                                                                    'lineHeight': '20px',
                                                                                    'borderWidth': '3px',
                                                                                    'borderStyle': 'solid',
                                                                                    'borderRadius': '5px',
                                                                                    'textAlign': 'center',
                                                                                    'margin': '25px 10px',
                                                                                    'padding': '20px'
                                                                                }
                                                                        )
                                                                ],
                                                     style = {'display': 'none'}
                                                    ),
                                            html.Div(id='output-data-upload'),
                                            dash_table.DataTable(id="data-preview"),
                                        ]    #end of children container for tab3
                                                     )    #end of container for tab3
                                       ]    #end of tab1 children
                        )    #end of tab1 arg
                    ]    #end of Tabs children
                )    # end of Tabs Arguments
            ]    #end of main html.Div children
        )    #end of html.Div arguments

# Update_callback function

# function to update the Outputs
def update_callback(change_tups):
    ''' input -->list of tuples--each tuple (position of output, change to be made).
        For handling long lists of outputs in a return statement
        eg. position 4 is Output(#5) for the options of the last name dropdown and change it to [] (empty options)
        would be (4, []) the remaining outputs would be no_update
        This is my favorite code I have ever written to make handling all the outputs so easy to manage!!'''
########Future: put in error message for debugging when the tup is out of range-- 
    ##########this means that the position within the tup does not match up the outputs_list  

    
    # Define default return values (all no_update)
    return_values = [no_update] * len(callback_context.outputs_list)       
    if change_tups:
        for change_tup in change_tups:
            # Modify only necessary values
            return_values[change_tup[0]] = change_tup[1]
    # Convert back to tuple and return
    return tuple(return_values)

#Tab1 upload functions

@app.callback(
########working here to tighten up the sheet selector and change the output
    # Output("sheet-selector-div", 'children'), #0
    Output("sheet-selector", "options"),  #0
    Output("sheet-selector", "value"),  #1
    Output("sheet-selector-div", "style"),  #2
    Output("df-storage", "data"),  #3
    Output('Error_message', 'children'),  #4
    Output('max-dict', 'data'),  #5
    Output('upload-H1', 'style'), #6
    Input("upload-data", "contents"),
    Input("df-storage", "data"),
    prevent_initial_call = True 
)
def process_upload(contents, stored_sheets):
    '''
    uploaded file will trigger this function (update is prevented if there is no uploaded file or 
    if there is already data in the storage, df-storage)
    contents of uploaded file is decoded and read into a pandas excel file object
    The data is then transferred into a dictionary of sheet_name : df for that sheet called stored sheets
    The first sheet name is set as the default sheet to use
    The data is then modified by
    '''

    # if there is stored data then upload has already been processed, if no contents then no file has been uploaded
    if contents and not stored_sheets:
        #Performance eval
        start_time = time.time()
        
        # uploaded file (contents) exist and no stored data
        #decode the uploaded file, create excel object
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        excel_data = io.BytesIO(decoded)
        xls = pd.ExcelFile(excel_data)  
        
        # Read all sheets into a dictionary
        stored_sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
        
        # Default sheet selection
        selected_sheet = xls.sheet_names[-1] if xls.sheet_names else None

        # set the sheet options based on the xls sheet names
        sheet_options = [{"label": sheet, "value": sheet} for sheet in stored_sheets.keys()]
        if len(xls.sheet_names) > 1:
            #if more than one sheet unhide the sheet dropdown
            sheet_toggle = {'display' : 'block'}
        else:
            # keep the dropdown hidden since there is only one sheet
            sheet_toggle = {'display' : 'none'}
    
        ### DF Modifications
        
        #modifications to the df before storage--loop through multiple sheets
        stored_sheets_modified = {}
        subtest_names = []
        for sheet, df in stored_sheets.items():
            if isinstance(df, pd.DataFrame):
                
                # Pull out the max values and store as a dictionary
                
                # the score is inferred from the dataset
                # get all the subtest column names loop thru and pull the max value from each column--
                # if there is a max_value row then that row will have the max value that can be inferred
                # then make the dictionary by zipping these two lists together
                # list all the date columns indices
                date_col_index = [df.columns.get_loc(col) for col in df.columns if 'date' in str(col).lower()]
                # using date indices as reference get column name next to date column
                subtest_columns = [df.columns[(index + 1)] for index in date_col_index]
                subtest_names.append(set(subtest_columns))    # this set of column names will be used to check for inconsistency
                if len(subtest_names) >= 2:
                    # check for inconsistencies between the sets of subtest names
                    # Compute symmetric difference across all sets
                    unique_columns = reduce(set.symmetric_difference, subtest_names)
                    if unique_columns:
                        
                        error_mess = ["🚨 Oops! There's a small issue with your spreadsheets. 🚨",
                                    "It looks like the column headers are not consistent across all your sheets.",
                                    "To avoid issues, first type the test(column) names in one sheet exactly how you want them,",
                                    "then copy and paste the column names into all the other sheets. This prevents small differences",
                                    "like capitalization or extra spaces from causing problems.",
                                    f"Just FYI, here are the offending column names {unique_columns}"
                                     ]
                        changes = [
                                    (4, [
                                    html.H3([
                                        error_mess[0], html.Br(),
                                        error_mess[1], html.Br(),
                                        error_mess[2], html.Br(),
                                        error_mess[3], html.Br(),
                                        error_mess[4], html.Br(),
                                        error_mess[5], html.Br(),
                                    ]
                                           )
                                ])
                                ]
                        return update_callback(changes)                      
                # loop thru subtest columns and pull max value from each column
                max_values = [df[col].max() for col in subtest_columns]
                
                #zip together the dictionary
                max_dict = {k:v for k,v in zip(subtest_columns, max_values)}

                # modify the df (Note: pandas auto changes dtype to date format for columns labeled date)
                df.rename(columns = {"Last Name": "Last_Name", "First Name": "First_Name", "Date": "Date.0"}, inplace = True)
                    
                #check for duplicate names
                #Modify the dataset to resolve ambiguity of duplicate names by adding teacher to first name
                df['Is_Duplicate'] = df.duplicated(subset=['First_Name', 'Last_Name'], keep=False)    #Note: keep = False marks all duplicates as True in the   is_duplicate column, not just the second as true
                # overwrite First_name column with unique identifiers
                df['First_Name'] = df.apply(
                                        lambda row: f"{row['First_Name']}({row['Teacher']})" if row['Is_Duplicate'] else row['First_Name'],
                                        axis=1
                                            )
            
                # After resolving duplicates by modifying first name, check again for duplicates
                # If still dups-- get names of dups and present error message with dups names for fixing
                if df.duplicated(subset=['First_Name', 'Last_Name'], keep=False).any():
                    # preview the names in the duplicates-df to tell the user which students need fixing
                    dups = df[df.duplicated(subset = ['First_Name', 'Last_Name', 'Teacher'], keep=False)]
                    dups = dups[['First_Name', 'Last_Name', 'Teacher']]
    
                    # Output 4 is a Div for error messages, 
                    changes = [
                        (4, [
                        html.H3(f'Unresolvable duplicates found. Please check spreadsheet for the following duplicates:'),
                        html.Pre(dups.to_string(index = False)),
                        html.H6("Refresh the page to upload a new file")
                    ])
                    ]
                    return update_callback(changes)
                    
                                
                # clean the data to make sure that all empty cells are NaN
                df = df.map(lambda x: pd.NA if isinstance(x, str) and x.strip() == '' else x)
            
                #check for categorical performance column
                # if the column name doesn't exist then check if it might be named something else
                if not "Performance Band RP2.1" in df.columns:
                    #check the df columns for the following values
                    cat_values = {"Above", "Benchmark", "Below", "Well Below"}
                    def find_categorical_column(df, values):
                        '''Function to detect the categorical column '''
                        for col in df.columns:
                            if df[col].astype(str).isin(values).any():  # Convert to string to handle mixed types
                                return col  # Return the first matching column
                        return None  # Return None if no matching column is found
                
                    # Check for the categorical column with the above function
                    cat_col = find_categorical_column(df, cat_values)
                    if cat_col:
                        #the column exists rename it "Performance Band RP2.1"
                        df.rename(columns = {cat_col: "Performance Band RP2.1"}, inplace = True)
                    else:
                        # the column doesn't exist--create a replacement column
                        # "No_Value" can be set in the dictionary as white background and white heart
                        df["Performance Band RP2.1"] = 'No_Value'
            
                #make a new column for the emojis to reference later
                performance_emojis = {'Above' :'💙', 'Benchmark' : '💚' , 'Below' : '💛' ,'Well Below' : '❤️', 'No_Value': '🤍'}   # emojis need to be treated as strings
                df['Emojis'] = df['Performance Band RP2.1'].map(performance_emojis)
                      
                #sort the df by performance
                performance_order = ['Well Below', 'Below', 'Benchmark', 'Above', 'No_Value']
                df['performance_category'] = pd.Categorical(df["Performance Band RP2.1"], categories = performance_order, ordered = True)
                df.sort_values(by = ['performance_category', 'First_Name'], inplace = True)
        
                #store the modified dataframe back into a new sheets_dictionary
                stored_sheets_modified.update({sheet : df})
            
               
        ######End of df modification
        
    
        # convert each modified sheet into a json file stored as a dictionary for storage into dcc.Store
        sheets_json = {sheet: df.to_json(orient = 'split') for sheet, df in stored_sheets_modified.items()}
       
        #Performance eval
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Upload Callback execution time: {elapsed_time:.3f} seconds")
                                             
        changes = [
            (0, sheet_options), 
            (1, selected_sheet),
            (2, {'display': "block"}),    #display sheet dropdown row
            (3, sheets_json),
            (5, max_dict),
            (6, {'display' : 'none'})    # hide main header(H1) on upload page
        ]
        return update_callback(changes)

    else:
        raise PreventUpdate

# update_table function

@app.callback(
    Output('Tab2','disabled'),                #0
    Output('output-data-upload', 'children'), #1   
    Output('upload-data', 'style'),           #2
    Input('df-storage', 'data'),
    Input('sheet-selector', 'value'),
    Input('max-dict', 'data'),
    State('upload-data', 'filename')
)
def update_table(stored_sheets, selected_sheet, max_dict, filename):
    '''Display the DataFrame as a preview in the dashboard 
       Check for duplicate names and resolve an ambiguity, error message for unresolvable ambiguity
       convert df to json for storage in dcc.Store for future use
       Note: can modify the df (add/subtract...) here before storage into Store
       unhide the other tab'''
#######Future: Update all documentation
    
        #pull the stored json data out of stored sheets and into a df
    if selected_sheet and stored_sheets:
         # Get selected sheet's data and create max value table to display above the data preview
        json_str = stored_sheets.get(selected_sheet)
        df = pd.read_json(io.StringIO(json_str), orient='split', convert_dates = False)
        
        # dash table preview of data and label that shows filename
        data_table = dash_table.DataTable(id = 'data-preview',
                                         columns = [{"name": col, "id": col} for col in df.columns],
                                         data = df.to_dict("records")
                                         )
        data_table_label = html.H3(f"Preview of uploaded file: {filename}, {selected_sheet}")
        
        # convert the max_dict to df (transpose to switch from one column to one row)
        max_df = pd.DataFrame.from_dict(max_dict, orient = 'index').T
        max_df.insert(0, '  ', ['max_value'])    # this labels the one row and keeps the column header blank for this labeling
        
        #make a dash table with this max_df to be output into output 3--html.div id = output-data-upload
        max_table = dash_table.DataTable(id="max-value-preview",
                                    columns = [{"name" : col, "id": col} for col in max_df.columns],
                                    data = max_df.to_dict("records")
                                        )
        #make a label for the max table                   )
        max_table_label = html.H3(f"Max values for each test:")        
        
        changes = [
            (0, False),                  # unhide Tab2
            (1, [max_table_label,
                 max_table,
                 html.Br(),
                 data_table_label,
                 data_table]),           # created list of children to be output into html.div
            (2, {'display': 'none'}),    # hide the drag and drop upload region
        ]
        return update_callback(changes)
    else:
        #if there is no stored sheets(nothing uploaded yet) display sample data 
        sample_dict = {
                    'Teacher': {0: 'Krummes', 1: 'Galen', 2: 'Dodd', 3: 'Sanders', 4: 'Santiago'},
                    'Grade': {0: 2, 1: 2, 2: 2, 3: 2, 4: 2},
                    'Student ID': {0: 'optional',
                              1: 'optional',
                              2: 'optional',
                              3: 'optional',
                              4: 'optional'},
                    'Last Name': {0: 'KiddR',
                                  1: 'KiddG',
                                  2: 'KiddM',
                                  3: 'KiddJ',
                                  4: 'KiddS',
                                  5: 'Max value',
                                 },
                    'First Name': {0: 'Sample',
                                    1: 'Sample', 
                                    2: 'Sample',
                                    3: 'Sample',
                                    4: 'Sample',
                                    5: ' '
                                  },
                    'Performance Band RP2': {0: 0, 1: 86, 2: 40, 3: 49, 4: 79, 5: 91},
                    'Performance Band RP2.1': {0: 'Well Below',
                                               1: 'Above',
                                               2: 'Below',
                                               3: 'Well Below',
                                               4: 'Benchmark'},
                    'Date.0': {0: np.nan, 1: '9/1/22', 2: '12/1/22', 3: '12/1/22', 4: '9/12/22'},
                    'test1...Any_Name ': {0: np.nan, 1: 18.0, 2: 20.0, 3: 15.0, 4: 20.0, 5: 21.0},
                    'Date.1': {0: np.nan, 1: '9/1/22', 2: '9/1/22', 3: '12/1/22', 4: '9/12/22'},
                    'test2...Short Vowel Sounds': {0: np.nan, 1: 5.0, 2: 4.0, 3: 5.0, 4: 3.0, 5: 5.0},
                    'Date.2': {0: np.nan, 1: '9/1/22', 2: '12/1/22', 3: '12/1/22', 4: '9/12/22'},
                    'etc...': {0: np.nan, 1: 5.0, 2: 3.0, 3: 5.0, 4: 5.0, 5 : 5.0},
                    'Date.13': {0: np.nan, 1: '12/1/22', 2: np.nan, 3: np.nan, 4: '9/1/22'},
                    'last test...3-5 Syllables': {0: np.nan, 1: 2.0, 2: np.nan, 3: np.nan, 4: 4.0, 5: 5}
                }
        sample_df = pd.DataFrame(sample_dict)
        sample_label = html.H3('Example data format')
        sample_table = dash_table.DataTable(id = 'sample_data',
                                           columns = [{"name": col, "id": col} for col in sample_df.columns],
                                            data = sample_df.to_dict("records")
                                           )
        changes = [
            (1, [sample_label,
                 sample_table,
                ]
            )
        ]
        return update_callback(changes)

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
     Input('TABS', 'value'),
     Input('reset-button', 'n_clicks'),
     Input('df-storage', 'data'),
     Input('sheet-selector', 'value'),
     Input('sheet-selector', 'options'),
     Input('max-dict', 'data')
    ]
)

def PieFig(full_name, tab, n_clicks, stored_sheets, selected_sheet, sheet_options, max_dict):
    '''When full name is chosen make pie fig using PieCharts function
    --
    '''  
    
## Pie chart function
    def PieCharts(student_data, current_sheet, max_dict):   
        # get all the subtest column names 

        # get column names from max_dict
        subtest_columns = [k for k, _ in max_dict.items()]

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
        sheet_name = f"{current_sheet}"
        title_text = f"{full_name}, {sheet_name}, Performance Band RP2 Overall:  {score}"
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
                color_map = {'Not Taken': 'grey', 'WIP': 'lightgrey'}
            else:
                #calculate incorrect using the subtest_column name and the max score dictionary
                incorrect = max_dict[col] - correct
                temp_df = {'Category': ['Yeah', 'WIP'], 'Count': [correct, incorrect]}
                color_map = {'WIP' : '#FF9999', 'Yeah' : '#228B22'}
                
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
            
            outline_color, outline_width = ('black', 3)
            fig.update_traces(marker=dict(line=dict(color=outline_color, width=outline_width)))
        
        return fig
## End of PieChart Function
        
    #logging time performance
    start_time = time.time()

    ######## In the case of only one sheet, sheet_options will be None--need to handle this case
    
    # when tab2(pie charts tab) is selected trigger function otherwise prevent update
    if tab == 'Tab2':
        if stored_sheets and selected_sheet:
            
             # Check the reset-button after stored and selected sheet present and before other parameters
                #reset-button click sets n_clicks to 1 this will reset the state parameters to stored_sheets = True, full_name = None
            
            if n_clicks > 0:
                n_clicks = 0    # reset the counter so as not to trigger again until clicked
                changes = [(0,{'display': 'block'}),    #unhide full name dropdown
                           (2, None),
                           (3, []),
                           (4, n_clicks),
                           (5, {'display': 'none'}),    #hide reset button
                           (6, ['Choose Kiddo'])
                          ]
                return update_callback(changes)
        
            if not full_name:
                #set the full_name_dropdown options
                # Get selected sheet's data

                if len(sheet_options) > 1:
                    #create empty set to collect all names and keep unique
                    student_names = set()
                    for sheet in sheet_options:
                        json_str = stored_sheets.get(sheet['value'])
                        df = pd.read_json(io.StringIO(json_str), orient='split', convert_dates = False)
                        student_names.update(zip(df['First_Name'], df['Last_Name']))
                    #once the student_names are collected then change them into the correct format for the fullname_options dropdown
                    fullname_options = [{'label': f'{first} {last}', 'value': f'{first}  {last}'} for first, last in student_names]

                else:
                    json_str = stored_sheets.get(selected_sheet)
                    df = pd.read_json(io.StringIO(json_str), orient='split', convert_dates = False)
                    fullname_options = [{'label': f'{first} {last} {heart}', 'value': f'{first}  {last}'} for first, last, heart in zip(df['First_Name'], df['Last_Name'], df['Emojis'])]
            
                changes = [
                    (1, fullname_options)
                ]
                return update_callback(changes)
            else:
                #full name present-- query the data

                
                # Get selected sheet's data
                json_str = stored_sheets.get(selected_sheet)
                df = pd.read_json(io.StringIO(json_str), orient='split', convert_dates = False)
    
                # get the record for the student
                first_name, last_name = full_name.split('  ')
                student_data = df.query("First_Name == @first_name & Last_Name == @last_name")
                # check that student is in the sheet selected, if not return an empty figure and message
                if student_data.empty:
                    changes = [
                        (6, [f"🙀 Oops! Looks like we couldn't find {first_name} {last_name} in {selected_sheet}. 🔎 Maybe check somewhere else?" ]),
                        (3, ['  '])    # return an empty figure
                    ]
                    return update_callback(changes)
        
                fig = PieCharts(student_data, selected_sheet, max_dict)
            
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
    else:
        raise PreventUpdate
server = app.server
# app.run_server(debug = True, port = 8044)



