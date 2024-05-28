
import json
import re
import numpy as np
import pandas as pd
from time import sleep
import httpx
from output_refine import output_refine_func
from llms import callLLM_hot, callLLM


def callLLMs(test_prompts, output_path, test_len=200, second=False, second_list=[], model_name='gpt-4'):
    print('begin calling APIs')
    if second:
        for i in second_list:
            output = callLLM_hot(test_prompts[i]['prompt'], model_name).choices[0].message.content
            test_prompts[i]['result'] = output
            sleep(0.5)
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
    with open(results_path+'.json', 'r') as f:
        content = f.read()
        test_prompts = json.loads(content)
    test_prompts = callLLMs(test_prompts, output_path, test_len, model_name=model_name)
    uncompliance = 0
    total = test_len
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
    print('begin experiments of ', results_path)
    for i in range(repeat_times):
        # compliance_rate = evaluate(results_dir+results_path, results_dir+results_path+'.json', test_len)
        compliance_rate = evaluate(results_dir+results_path, results_dir+'results_'+model+'_seed'+str(i)+'_'+results_path+'.json', test_len, model_name=model_name)
        row = [compliance_rate, results_path, i, model]
        df.loc[len(df)] = row
        df.to_csv('./log/compliance_rate.csv', index=False)

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

    results_path = 'test_history10_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_shuffle_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history15_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)
    
    results_path = 'test_history20_candidate20_shuffle_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_profile_gpt3.5_history_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_profile_gpt3.5_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_profile_gpt4_history_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_profile_gpt4_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_profile_glm4_history_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)

    results_path = 'test_history10_candidate20_profile_glm4_sample'
    df = repeat_exp(results_path, results_dir, df, model, repeat_times, test_len, model_name=model_name)



    
