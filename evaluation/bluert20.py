from bleurt import score

## Download the bluert-20 checkpoint and store locally
checkpoint = "C:/Users/weijie.tay.2019/Documents/GitHub/medicalbot/BLEURT-20"
references = ["apple is fresh"]
candidates = ["apple is fresh"]

## Load the bluert scorer
scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)
assert type(scores) == list and len(scores) == 1
print(scores)

import pandas as pd
import os

## Load the csv containg predictions
directory = "C:/Users/weijie.tay.2019/Documents/GitHub/medicalbot/test_data/evaluations/test_set_3_lasseEdits"
# directory = "C:/Users/weijie.tay.2019/Documents/GitHub/medicalbot/test_data/faiss/ratios"
# o_directory = "C:/Users/weijie.tay.2019/Documents/GitHub/medicalbot/test_data/faiss/eval_ratios"
for filename in os.listdir(directory):
    if(filename.endswith('model_save_GPT-med_MEDBERT_Lasse_15.csv')):
        print(filename)
        df = pd.read_csv(os.path.join(directory, filename))
        df['bleurt20_gpt'] =  df.apply (lambda row: scorer.score(references=[row['actual']], candidates=[row['pred']])[0], axis=1)
        # df['bleurt20_faiss'] =  df.apply (lambda row: scorer.score(references=[row['actual']], candidates=[row['pred_faiss']])[0], axis=1)
        
        ## View bluert score stats
        print(df['bleurt20_gpt'].describe())
        # print(df['bleurt20_faiss'].describe())
        df.to_csv(os.path.join(directory, filename), index=False)