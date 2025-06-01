import re
import json

def resultsMap(title_str):
    if type(title_str) == list:
        title_str = title_str[0]
    title_str = title_str.replace("\\'", "\'")
    title_str = title_str.replace("®", "&reg;")
    title_str = title_str.replace("è", "&egrave;")
    title_str = title_str.replace("★", "&#9733;")
    # title_str = title_str.replace("\\\\", "\\")
    return title_str

def output_refine_func(results_str_list, test_len, target_len):
    # ChatGPT & ChatGLM
    results = []
    invalid_format = []
    result_str = ''
    for i in range(test_len):
        try:
            # Extract the string between the first and last square brackets
            if len(re.findall(r'\[[\s\S]*\]', results_str_list[i])) == 1:
                result_str = re.findall(r'\[[\s\S]*\]', results_str_list[i])[0]
            else:
                raise ValueError
            # result_str = a[i]['result'][begin_idx[0]:end_idx[1]+1]
            # Remove comments and extra spaces
            result_str = re.sub(r'\'[,]*[ \n\r]+#.*$', '\',', result_str, flags=re.MULTILINE)
            result_str = re.sub(r'\'[ \n\r]*,[ \n\r]*\'', '\", \"', result_str.replace('\"', '\''))
            result_str = re.sub(r'\'[ \n,]*\]', '\"]', result_str)
            result_str = re.sub(r'\[[ \n]*\'', '[\"', result_str)
            result_str = result_str.replace('[\'', '[\"').replace('\\', '\\\\')
            results.append(json.loads(result_str))
        except:
            invalid_format.append(i)
            print(i)
            print(result_str)
    print('check format: ', invalid_format)

    # Check length
    invalid_list = []
    for i in range(len(results)):
        if len(results[i]) != target_len:
            print(i)
            invalid_list.append(i)
            print('length: ', len(results[i]))
            if len(results[i]) > target_len:
                results[i] = results[i][:target_len]
            else:
                results[i] = results[i] + ['N\\A']*(target_len-len(results[i]))
    print('check length: ', invalid_list)
    
    # Check compliance rate
    compliance_rate = 1-(len(invalid_format)+len(invalid_list))/test_len
    print('compliance rate: ', compliance_rate)
    results = [list(map(lambda x: resultsMap(x), entry)) for entry in results]
    return len(invalid_format)+len(invalid_list), results, invalid_format