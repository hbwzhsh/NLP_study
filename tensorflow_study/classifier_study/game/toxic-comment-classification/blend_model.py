import pandas as pd
import numpy as np

cnn_gru = pd.read_csv("submission_record/submission_cnn_gru_glove.csv")
lstm= pd.read_csv("submission_record/submission_lstm_fasttex.csv")
lr = pd.read_csv("submission_record/submission_tf-idf_nblr_3.csv")

b1 = cnn_gru.copy()
col = cnn_gru.columns

col = col.tolist()
col.remove('id')
print('Blending modle!!!')

# blend models.
# calulation final score acorrding to the 3 models scores.
score_0 = 0.986
score_1 = 0.987
score_2 = 0.982

for i in col:
    b1[i] = (score_0 * cnn_gru[i]  + score_1 * lstm[i] + score_2 * lr[i] ) / (score_0 + score_1 + score_2)

# submission files output 
b1.to_csv('submission_record/blend_submission.csv', index = False)

print("Blend over and ouput submission file!!!")
