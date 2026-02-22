'''
Adapted from https://github.com/lupantech/ScienceQA
'''

import os
import torch
import json
import argparse
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
from evaluations import caculate_bleu, caculate_rouge, caculate_similariry

warnings.filterwarnings('ignore')

def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_data, rationale_data, results_reference, data_file):
    results = result_data
    # num = len(results) # num should be the number of samples actually participating in the scoring, which will be defined later
    # assert num == 4241 # This assertion needs to be commented out when you run a small subset

    sqa_data = json.load(open(data_file))
    sqa_pd = pd.DataFrame(sqa_data).T
    full_test_res_pd = sqa_pd[sqa_pd['split'] == 'test'].copy()

    processed_qids_indices = []  # An index used to collect the rows that have actually been processed

    for index, row in full_test_res_pd.iterrows():
        if str(index) not in results:
            continue

        processed_qids_indices.append(index)  # Add to list

        full_test_res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        full_test_res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        full_test_res_pd.loc[index, 'has_image'] = True if row['image'] else False
        full_test_res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[str(index)])  # use str(index)
        full_test_res_pd.loc[index, 'pred'] = pred
        full_test_res_pd.loc[index, 'true_false'] = (label == pred)

    # Create a DataFrame that contains only processed samples for scoring
    if not processed_qids_indices:  # If none of the matching QIDs exist
        print("WARNING: No QIDs from predictions were found in the test data. Returning zero scores.")
        # You need to define a logic that returns a dictionary of all-zero scores
        return {"answer": {k: "0.00" for k in
                           ['acc_natural', 'acc_social', 'acc_language', 'acc_has_text', 'acc_has_image',
                            'acc_no_context', 'acc_grade_1_6', 'acc_grade_7_12', 'acc_average']},
                "rationale": {k: 0.0 for k in ['bleu1', 'bleu4', 'rouge', 'similariry']}}

    res_pd_for_scoring = full_test_res_pd.loc[processed_qids_indices]
    num_scored_samples = len(res_pd_for_scoring)  # This is the number of samples that actually participated in the scoring
    print(f"INFO: Number of results actually scored (found in predictions and test data): {num_scored_samples}")

    # accuracy scores - Based on res_pd_for_scoring calculations
    if num_scored_samples > 0:
        acc_average = len(res_pd_for_scoring[res_pd_for_scoring['true_false'] == True]) / num_scored_samples * 100
    else:
        acc_average = 0.0

    # Rationale quality calculation, this part is not affected
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()  # Make sure to load only once
    bleu1 = caculate_bleu(rationale_data, results_reference, gram=1)
    bleu4 = caculate_bleu(rationale_data, results_reference, gram=4)
    rouge = caculate_rouge(rationale_data, results_reference)
    similariry = caculate_similariry(rationale_data, results_reference, model)

    scores = {
        "answer": {  # All get_acc_with_contion here should use res_pd_for_scoring
            'acc_natural':
                get_acc_with_contion(res_pd_for_scoring, 'subject', 'natural science'),
            'acc_social':
                get_acc_with_contion(res_pd_for_scoring, 'subject', 'social science'),
            'acc_language':
                get_acc_with_contion(res_pd_for_scoring, 'subject', 'language science'),
            'acc_has_text':
                get_acc_with_contion(res_pd_for_scoring, 'has_text', True),
            'acc_has_image':
                get_acc_with_contion(res_pd_for_scoring, 'has_image', True),
            'acc_no_context':
                get_acc_with_contion(res_pd_for_scoring, 'no_context', True),
            'acc_grade_1_6':
                get_acc_with_contion(res_pd_for_scoring, 'grade',
                                     ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
            'acc_grade_7_12':
                get_acc_with_contion(res_pd_for_scoring, 'grade',
                                     ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
            'acc_average':
                "{:.2f}".format(acc_average),  # Use the correctly calculated acc_average above
        },
        "rationale": {
            'bleu1': bleu1 * 100,
            'bleu4': bleu4 * 100,
            'rouge': rouge * 100,
            'similariry': similariry * 100,
        }
    }
    del model  # Free up the video memory occupied by the SentenceTransformer model
    torch.cuda.empty_cache()  # Clean the CUDA cache
    return scores


# Make sure that get_acc_with_contion also has zero protection
def get_acc_with_contion(res_pd_to_score, key, values):
    if isinstance(values, list):
        total_pd = res_pd_to_score[res_pd_to_score[key].isin(values)]
    else:
        total_pd = res_pd_to_score[res_pd_to_score[key] == values]

    # print(f"DEBUG (get_acc_with_contion): For condition '{key}'=='{values}', found {len(total_pd)} samples.") # 可以保留这个打印

    if len(total_pd) == 0:  # Zero protection
        return "0.00"

    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)
