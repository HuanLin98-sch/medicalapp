# Translation

### `translate.py`

1.  Make sure you have a `./data/raw_data` and `./data/translated/data` folder
2.  Place the untranslated files in `raw_data` folder
3.  Select a model on `model = EasyNMT(<model name>)`
4.  Run the code with `python3 translate.py`



### `evaluate.py`

1.  Make sure you have a `./data/raw_data` and `./data/translated/data` folder
2.  Place the dictionary in `raw_data` folder
3.  Pre-process the dictionary
    -   Uncomment Step 1 of the code (Comment the rest out) and run with `python3 evaluate.py`
4.  Run the translation models part to translate the dictionary
    -   Uncomment Step 2 to Step 4 of the code (Comment the rest out) and run with `python3 evaluate.py`
5.  Run the spaCy vector similarity part to calculate the word similarity
    -   Uncomment Step 5 of the code (Comment the rest out) and run with `python3 evaluate.py` 
6.  Use Excel to view the final csv file and calculate the average word similarity for each model 