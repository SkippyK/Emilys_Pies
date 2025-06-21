# Emilys_Pies
Custom Data Visualization tool

This project came out of two things happening simultaneously, 1) An online class to learn dashboards using plotly and dash with Udemy taught by Chris Bruehl and 2) my dear friend Emily needing some way to visual her spreadsheet data in a useful way. After some convincing she asked permission to give me some anonymized data such that I could create a project using what I had learned from the Dashboard class. I felt that the best way for my friend Emily to use this program was to try and host it on the web via pythonanywhere.com, to avoid the issue of trying to configure her personal computer.  This github repository is the final project and is being web hosted at SkippyK.pythonanywhere.com

The program consists of two tabs, the first default tab is an upload page, the second tab is where the pie charts will be displayed after the spreadsheet data has been processed. It is important to note that since the spreadsheet data has some personal information I wrote the code such that the spreadsheet is turned into a pandas dataframe and only held in memory for as long as the user is using it, it being--the dataframe, and the webpage. When the webpage is refreshed or the browser is closed the dataframe and any spreadsheet data is gone. After the data is uploaded, similar to how spreadsheet data is uploaded to google the pie tab will be available. After choosing the pie tab a dropdown menu appears to choose which person you want to choose. After choosing a name a figure is created that will present a pie chart for each of the tests taken by that person.
The program only works if the spreadsheet (.xls) has a specific order of column headers, for example:
random column headers...First Name,  Last Name,  Score, Category,  Date,  Test1,  Date,  Test2,  Date,  Test3...

The order of the column headers and the Date columns is what the program uses to identify the relevant info, this eliminates the need to know what the actual column headers say, due to some inconsistency in the column headers between spreadsheets.
I have included a sample data csv file to be able to use the app as intended. 

I am continuing to refine this project and make it more robust to handle more diverse formats, however version 2 is currently the version that is being webhosted.

I take seriously the security of any information that might be considered private and have taken steps, to the best of my ability, to minimize access to the data uploaded to this program. The uploaded data is not retained beyond the scope of the program and is no longer accessible once the web page is closed.



