# Bay-Area-Food-Access

To run this analysis, all the provided datasets have already been retrieved and stored in the data directory. Simply run the included midterm.ipynb file.

If desiring to rerun the Google Maps harvest of stores again, the user must provide their own Google Maps API key. However, previously harvested data is already included in the data directory.

To re-run the census data API call, the user must also provide their own US Census Bureau API key. However, the data is already included in the data directory as ‘selected_census_data.csv’ & 'cultural_food_census.csv'. Instructions and general format for API call code is in 'census_api_call_instructions.txt'

To determine what the store generally provides, we need to know which term/part of the name signifies what classifier. Some examples have already been provided. However, the list for each classifier does not consist of all possible. There is code already that provides a sample of the possible types and parts of name that you can repeatedly run to manually label and enter them into the list for each classifier. 

