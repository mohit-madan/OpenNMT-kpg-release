import pandas as pd
import os
import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import math
import random


# input format of excel files should be correct otherwise wrong json will be created.
def main(filedir, sheet=None):
    random.seed(8)
    dr = os.getcwd() + "\\" + filedir
    filesList = [dr + "\\" + f for f in listdir(dr) if isfile(join(dr, f))]

    finalDF = pd.DataFrame()

    for file in tqdm(filesList):
        if file.lower().endswith(('.xls', 'xlsx')):
            if sheet:
                data_df = pd.read_excel(file, sheet_name=sheet)
            else:
                data_df = pd.read_excel(file, sheet_name=0)

        elif file.lower().endswith('.csv'):
            data_df = pd.read_csv(file)
        else:
            pass
        data_df = data_df.dropna(how='all')
        data_df = data_df.reset_index(drop=True)
        codeCols = [ind for ind, x in enumerate(list(data_df)) if "code" in x.lower()]
        remainCols = [ind for ind, x in enumerate(list(data_df)) if "code" not in x.lower()]
        codesDF = data_df.iloc[:, codeCols]
        remainDF = data_df.iloc[:, remainCols]


        #######################################
        # add separator and replace NaN to empty space
        # convert to lists
        arr = codesDF.add('/').fillna('').values.tolist()
        # list comprehension, replace empty spaces to NaN
        combinedCodes = pd.Series([''.join(x).strip('/') for x in arr]).replace('^$', np.nan, regex=True)
        # replace NaN to None
        combinedCodes = combinedCodes.where(combinedCodes.notnull(), None)
        combinedCodes.name = "keywords"
        ##################################################
        newDF = remainDF.join(combinedCodes)
        newDF['keywords'] = newDF.keywords.apply(lambda x: str(x).split('/'))
        allKeywords = set(newDF['keywords'].apply(pd.Series).stack().reset_index(drop=True).unique())
        newDF['allKeywords'] = [list(allKeywords - set(newDF['keywords'].iloc[i])) for i in newDF.index]
        finalDF = finalDF.append(newDF, ignore_index=True)

    finalDF = finalDF.rename(columns= {'Src': "abstract"}, inplace=False)

    # shuffle pd rows randomly
    finalDF.sample(frac=1)

    finalDF = finalDF.reset_index(drop=True)
    finalDF['id'] = finalDF.index
    finalDF['title']= ""
    finalDF['id'] = finalDF['id'].astype("|S")
    totalRows = len(finalDF.index)

    train_df = finalDF.iloc[0:math.floor(totalRows*0.7), :]
    validate_df = finalDF.iloc[math.floor(totalRows*0.7):math.floor(totalRows*0.8), :]
    test_df = finalDF.iloc[math.floor(totalRows*0.8):, :]

    train_json = train_df.to_json(orient="records", lines=True)
    validate_json = validate_df.to_json(orient="records", lines=True)
    test_json = test_df.to_json(orient="records", lines=True)

    train_file = (os.path.splitext(filedir)[0] + "_" + sheet + ".json" if sheet else os.path.splitext(filedir)[
        0]) + "_train" + ".json"
    validate_file = (os.path.splitext(filedir)[0] + "_" + sheet + ".json" if sheet else os.path.splitext(filedir)[
        0]) + "_validate" + ".json"
    test_file = (os.path.splitext(filedir)[0] + "_" + sheet + ".json" if sheet else os.path.splitext(filedir)[
        0]) + "_test" + ".json"

    with open(train_file, 'w') as f:
        f.write(train_json)
    with open(validate_file, 'w') as f:
        f.write(validate_json)
    with open(test_file, 'w') as f:
        f.write(test_json)

    # with jsonlines.open('output.jsonl', 'w') as writer:
    #     writer.write_all(json_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--sheet", "-t", type=str)
    args = parser.parse_args()

    main(args.file, args.sheet)