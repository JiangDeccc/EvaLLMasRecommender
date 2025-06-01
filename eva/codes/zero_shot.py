
import json
import re
import numpy as np
import pandas as pd
from time import sleep
import httpx
from output_refine import output_refine_func
from llms import callLLM_hot, callLLM


def callLLMs(test_prompts, output_path, test_len=200, second=False, second_list=[], model_name='gpt-4'):
    """
    Call LLMs to get the results for the test prompts.
    Args:
        test_prompts (list): List of test prompts.
        output_path (str): Path to save the results.
        test_len (int): Number of test prompts to process.
        second (bool): Whether to call the LLMs for the second time.
        second_list (list): List of indices to call the LLMs for the second time.
        model_name (str): Name of the model to use for calling LLMs.
    Returns:
        list: Updated test prompts with results.
    """
    print('begin calling APIs')
    # Check if it is the second call
    # If second is True, we only call the LLMs for the indices in second_list
    if second:
        for i in second_list:
            output = callLLM_hot(test_prompts[i]['prompt'], model_name).choices[0].message.content
            test_prompts[i]['result'] = output
            sleep(0.5)
    # If second is False, we call the LLMs for all prompts
    else:
        for i in range(test_len):
            if test_prompts[i].get('result', ''):
                continue
            output = callLLM(test_prompts[i]['prompt'], model_name).choices[0].message.content
            test_prompts[i]['result'] = output
            sleep(0.5)
            if i%10 == 0:
                print('current step: ', i/test_len)
            if i%10 == 0 and i != 0:
                json_file = open(output_path, mode='w')
                json.dump(test_prompts, json_file, indent=4) 
    json_file = open(output_path, mode='w')
    json.dump(test_prompts, json_file, indent=4) 
    print('finished!')
    return test_prompts

def evaluate(results_path, output_path, test_len=200, model_name='gpt-4'):
    """
    Evaluate the compliance rate of the results.
    Args:
        results_path (str): Path to the results file.
        output_path (str): Path to save the output results.
        test_len (int): Number of test prompts to process.
        model_name (str): Name of the model to use for calling LLMs.
    Returns:
        float: Compliance rate.
    """
    with open(results_path+'.json', 'r') as f:
        content = f.read()
        test_prompts = json.loads(content)
    test_prompts = callLLMs(test_prompts, output_path, test_len, model_name=model_name)
    uncompliance = 0
    total = test_len
    # Refine the output and calculate compliance rate
    while True:
        invalid_num, results, invalid = output_refine_func([x['result'] for x in test_prompts[:test_len]], test_len, 10)
        uncompliance += invalid_num
        total += len(invalid)
        if len(invalid):
            test_prompts = callLLMs(test_prompts, output_path, test_len, second=True, second_list=invalid, model_name=model_name)
        else:
            break
    json_file = open(output_path, mode='w')
    json.dump(test_prompts, json_file, indent=4)
    return 1-uncompliance/total

def repeat_exp(results_path, results_dir, df, model, repeat_times, test_len=200, model_name='gpt-4'):
    """
    Repeat the experiments for different results paths and save the compliance rates.
    Args:
        results_path (str): Path to the results file.
        results_dir (str): Directory where the results are stored.
        df (DataFrame): DataFrame to store the compliance rates.
        model (str): Model name.
        repeat_times (int): Number of times to repeat the experiments.
        test_len (int): Number of test prompts to process.
        model_name (str): Name of the model to use for calling LLMs.
    Returns:
        DataFrame: Updated DataFrame with compliance rates.
    """
    print('begin experiments of ', results_path)
    for i in range(repeat_times):
        # compliance_rate = evaluate(results_dir+results_path, results_dir+results_path+'.json', test_len)
        compliance_rate = evaluate(results_dir+results_path, results_dir+'results_'+model+'_seed'+str(i)+'_'+results_path+'.json', test_len, model_name=model_name)
        row = [compliance_rate, results_path, i, model]
        df.loc[len(df)] = row
        df.to_csv('./log/compliance_rate.csv', index=False)

    # Add empty rows to the DataFrame
    for i in range(3):
        row = [''] * 4
        df.loc[len(df)] = row
        df.to_csv('./log/compliance_rate.csv', index=False)
    print('end experiments of ', results_path)
    return df

if __name__ == '__main__':
    repeat_times = 1
    test_len = 200
    dataset = "beauty"
    model = 'gpt4'
    model_name = 'gpt-4'
    columns = ['compliance rate', 'path', 'random_seed', 'model']
    df = pd.DataFrame(columns=columns)
    results_dir = f'./results/{dataset}/'

    # history in temporal order
    results_path = 'test_history10_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # history in random order
    results_path = 'test_history10_shuffle_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # positive candidate at the first
    results_path = 'test_history10_candidate20_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # history length = 15
    results_path = 'test_history15_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)
    
    # history length = 20
    results_path = 'test_history20_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # with gpt3.5 profile + history
    results_path = 'test_history10_candidate20_profile_gpt3.5_history_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # with gpt3.5 profile
    results_path = 'test_history10_candidate20_profile_gpt3.5_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # with gpt4 profile + history
    results_path = 'test_history10_candidate20_profile_gpt4_history_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # with gpt4 profile
    results_path = 'test_history10_candidate20_profile_gpt4_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # with glm4 profile + history
    results_path = 'test_history10_candidate20_profile_glm4_history_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    # with glm4 profile
    results_path = 'test_history10_candidate20_profile_glm4_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)



    
