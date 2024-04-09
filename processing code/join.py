import pandas as pd
import glob, os
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('stsb-roberta-large')



if __name__ == '__main__':
    files = glob.glob("test_final/*.csv")
    tmp = []
    for i in files:
        df = pd.read_csv(i, lineterminator='\n')
        df = df[['topic', 'sentiment']]
        filename = i.replace('test_final/', '').replace('.csv', '')
        df.rename(columns={'topic':'topics', 'sentiment':filename}, inplace=True)
        tmp.append(df)
    tmp = pd.concat(tmp)
    tmp = tmp.groupby('topics').first()
    tmp = tmp.reset_index()
    tmp[['3_16', '4_16', '5_16', '6_16', '7_16', '8_16', '9_16', '10_16', '11_16', '12_16', '3_17', '4_17', '5_17', '6_17', '7_17', '8_17', '9_17', '10_17', '11_17', '12_17', '3_18', '4_18', '5_18', '6_18', '7_18', '8_18', '9_18', '10_18', '11_18', '12_18', '3_19', '4_19', '5_19', '6_19', '7_19', '8_19', '9_19', '10_19', '11_19', '12_19', '3_20', '4_20', '5_20', '6_20', '7_20', '8_20', '9_20', '10_20', '11_20', '12_20', '3_21', '4_21', '5_21', '6_21', '7_21', '8_21', '9_21', '10_21', '11_21', '12_21']] = 'NaN'
    tmp = tmp[['topics', 
'1_16', 
'2_16', 
'3_16', 
'4_16', 
'5_16', 
'6_16', 
'7_16', 
'8_16', 
'9_16', 
'10_16', 
'11_16', 
'12_16', 
'1_17', 
'2_17', 
'3_17', 
'4_17', 
'5_17', 
'6_17', 
'7_17', 
'8_17', 
'9_17', 
'10_17', 
'11_17', 
'12_17', 
'1_18', 
'2_18', 
'3_18', 
'4_18', 
'5_18', 
'6_18', 
'7_18', 
'8_18', 
'9_18', 
'10_18', 
'11_18', 
'12_18', 
'1_19', 
'2_19', 
'3_19', 
'4_19', 
'5_19', 
'6_19', 
'7_19', 
'8_19', 
'9_19', 
'10_19', 
'11_19', 
'12_19', 
'1_20', 
'2_20', 
'3_20', 
'4_20', 
'5_20', 
'6_20', 
'7_20', 
'8_20', 
'9_20', 
'10_20', 
'11_20', 
'12_20', 
'1_21', 
'2_21', 
'3_21', 
'4_21', 
'5_21', 
'6_21', 
'7_21', 
'8_21', 
'9_21', 
'10_21', 
'11_21', 
'12_21'
]]

    tmp.to_csv('test_tmp.csv', index=False)

