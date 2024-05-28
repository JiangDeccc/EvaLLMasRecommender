import argparse
import json
import numpy as np 
import pandas as pd
import re
from collections import Counter

def parse_global_args(parser):
    parser.add_argument('--dataset', type=str, default='beauty',
                        help='The dataset to be examined.')
    parser.add_argument('--results_path', type=str, default='../../llm_results/beauty/test_history10_candidate20_shuffle_sample',
                        help='Recommended lists file path.')
    parser.add_argument('--sota', type=bool, default=False,
                        help='To test SOTA models\' or LLM models\' results.')
    parser.add_argument('--model_name', type=str, default='BPRMF',
                        help='SOTA model to be evaluated.')
    parser.add_argument('--metric_len', type=int, default=5,
                        help='Top-K recommendation.')
    parser.add_argument('--sample_num', type=int, default=200,
                        help='The amount of sampling users.')
    parser.add_argument('--save', type=bool, default=True,
                        help='To save the calculation results or not.')
    parser.add_argument('--history_max', type=int, default=5,
                        help='The history max length of sequential models.')
    parser.add_argument('--shuffle_history', type=bool, default=False,
                        help='Whether to random shuffle the user history or not.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed.')    
    parser.add_argument('--cand_len', type=int, default=20,
                        help='The amount of candidate items.')    
    return parser

def read_list(path):
    test_samples = []
    file = open(path, 'r')
    for line in file.readlines():
        test_samples.append(line[:-1])
    file.close()
    test_samples = [x.split(' ') for x in test_samples]
    return test_samples

def read_meta(dataset):
    f = open(f'../../data/{dataset}/datamaps.json', 'r')
    content = f.read()
    data_maps = json.loads(content)
    f.close()
    # item meta
    file = open(f"../../data/{dataset}/meta.json", 'r')
    metas = json.load(file)
    file.close()
    metas = metas.split('\n')[:-1]
    metas = [eval(x) for x in metas]
    for item in metas:
        try:
            _ = (item['asin'], item['title'])
        except:
            item['title'] = 'N\A'
    item_asin2name = [(x['asin'], x['title']) for x in metas]
    item_asin2name = {key: value for key, value in item_asin2name}
    # name2item_asin = [(x['title'], x['asin']) for x in metas]
    # name2item_asin = {key: value for key, value in name2item_asin}
    name2item_id = [(item_asin2name[data_maps['id2item'][x]], x) for x in data_maps['id2item'].keys()]
    name2item_id = {key: value for key, value in name2item_id}
    id2categories = {int(name2item_id.get(x['title'], '0')): x['categories'][0] for x in metas}
    return name2item_id, id2categories

def resultsMap(title_str):
    if type(title_str) == list:
        title_str = title_str[0]
    title_str = title_str.replace("\\'", "\'")
    title_str = title_str.replace("®", "&reg;")
    title_str = title_str.replace("è", "&egrave;")
    title_str = title_str.replace("★", "&#9733;")
    # title_str = title_str.replace("\\\\", "\\")
    return title_str

def create_pattern(target):
    target_adjusted = re.escape(target)
    target_adjusted = re.sub(r'&(?!amp;)', '&amp;', target_adjusted)
    target_adjusted = target_adjusted.replace("'", "(?:\'|&quot;)").replace('&quot;', "(?:\'|&quot;)")
    pattern = re.compile(target_adjusted, re.IGNORECASE)
    return pattern

def mapTitle2itemid(title, name2item_id, name_str):
    item_id = int(name2item_id.get(title, '0'))
    if item_id == 0:
        pattern = create_pattern(title)
        match = re.search(pattern, name_str)
        matched_substring = match.group() if match else "no"
        # if not match:
        #     print(title)
        return int(name2item_id.get(matched_substring, '0'))
    else:
        return item_id

def mapListTitle2item_id(title_list, name2item_id, name_str):
    try:
        return [mapTitle2itemid(x, name2item_id, name_str) for x in title_list]
    except:
        print(title_list)
        for item in title_list:
            print(mapTitle2itemid(item, name2item_id, name_str))


def read_llm_results(results_path, name2item_id, name_str, test_len, model=None, cand_len=20):
    # input format
    # 'result': recommended list
    # 'target': item title
    # 'user': user id
    f = open(results_path+'.json', 'r')
    content = f.read()
    a = json.loads(content)
    f.close()

    err = False
    if model=='llama':
    # Llama
        results = []
        invalid_format = []
        result_str = ''
        for i in range(test_len):
            try:
                # print(len(re.findall(r'[0-9]+\. .*', a[i]['result'])))
                if len(re.findall(r'[0-9]+\. .*', a[i]['result'])) == 10:
                    result_str = re.findall(r'[0-9]+\. .*', a[i]['result'])
                else:
                    raise ValueError
                result_str = list(map(lambda x: re.sub(r'[0-9]+\. ', '', x), result_str))
                results.append(result_str)
            except:
                results.append(['N\A']*10)
                invalid_format.append(i)
                err = True
                print(i)
                print(a[i]['result'])
                # err_string = result_str
        print(invalid_format)

    else:
        # ChatGPT & ChatGLM
        results = []
        invalid_format = []
        result_str = ''
        for i in range(test_len):
            try:
                if len(re.findall(r'\[[\s\S]*\]', a[i]['result'])) == 1:
                    result_str = re.findall(r'\[[\s\S]*\]', a[i]['result'])[0]
                else:
                    # result_str = '[]'
                    raise ValueError
                # result_str = a[i]['result'][begin_idx[0]:end_idx[1]+1]
                result_str = re.sub(r'\'[,]*[ \n\r]+#.*$', '\',', result_str, flags=re.MULTILINE)
                result_str = re.sub(r'\'[ \n\r]*,[ \n\r]*\'', '\", \"', result_str.replace('\"', '\''))
                result_str = re.sub(r'\'[ \n,]*\]', '\"]', result_str)
                result_str = re.sub(r'\[[ \n]*\'', '[\"', result_str)
                result_str = result_str.replace('[\'', '[\"').replace('\\', '\\\\')
                results.append(json.loads(result_str))
            except:
                invalid_format.append(i)
                print(i)
                print(a[i]['result'])
                err = True
        print(invalid_format)
    if err:
        raise ValueError

    print('check length')
    invalid_list = []
    for i in range(len(results)):
        if len(results[i]) != 10:
            print(i)
            invalid_list.append(i)
            print('length: ', len(results[i]))
            if len(results[i]) > 10:
                results[i] = results[i][:10]
            else:
                results[i] = results[i] + ['N\A']*(10-len(results[i]))
    print(invalid_list)
    print('compliance rate')
    print(1-(len(invalid_format)+len(invalid_list))/test_len)

    results = [list(map(lambda x: resultsMap(x), entry)) for entry in results]
    targets = list(map(lambda x: x['target'], a[:test_len]))
    id_results = [mapListTitle2item_id(x, name2item_id, name_str) for x in results]
    id_targets = list(map(lambda x: name2item_id[x], targets))
    return id_results, id_targets

def read_sota_results(dataset, model, test_samples, sample, history_max, shuffle, random_seed, cand_len=20):
    df = pd.read_csv(f'../../sota_results/{dataset}/rec-{model}-history{history_max}-shuffle{shuffle}-seed{random_seed}.csv', sep='\t')
    results_dict = df.set_index('user_id')['rec_items'].to_dict()
    id_results = list(map(lambda x: eval(results_dict[x]), sample))
    id_targets = list(map(lambda x: test_samples[x-1][-1], sample))
    cnt = 0
    for i in range(len(id_results)):
        if len(id_results[i]) > cand_len:
            id_results[i] = id_results[i][:cand_len]
        elif len(id_results[i]) < cand_len:
            id_results[i] = id_results[i] + [0]*(cand_len-len(id_results[i]))
        else:
            cnt += 1
    print('compliance rate:', cnt/test_len)
    return id_results, id_targets


# MostPop Baseline
def generate_mostpop(test_samples, negative_samples, sample, cand_len=20, sample_cand_list=[]):
    pop_result = []
    acc_result = []
    cand_list = []
    for i in range(test_len):
        if len(sample_cand_list) == 0:
            cand_len_list = [test_samples[sample[i]-1][-1]] + negative_samples[sample[i]-1][1:cand_len]
        else:
            cand_len_list = sample_cand_list[i]
        cand_freq_list = np.array(list(map(lambda x: train_id_freq_dict.get(x, 0), cand_len_list)))
        idx_list = np.argsort(-cand_freq_list)
        acc_result.append(np.where(idx_list == 0)[0][0]+1)
        # if i == 0:
        #     print(idx_list)
        recommend_list = [cand_len_list[idx_list[i]] for i in range(10)]
        pop_result.append(recommend_list)
        cand_list.append(cand_len_list)
    hit_pop = np.array(list(map(lambda x: int(x>0 and x<=metric_len), acc_result)))
    return pop_result, acc_result, hit_pop, cand_list


# metrics
# accuracy
def calHR(hit):
    return np.array(hit).mean()

def calNDCG(hit, rank):
    return (np.array(hit) / np.log2(np.array(rank)+1)).mean()

def calMRR(hit, rank):
    return (np.array(hit)/np.array(rank)).mean()

# popularity bias
def calARP(freq_dict, id_results, metric_len):
    results_freq = np.array(list(map(lambda x: freq_dict.get(str(x), 0), np.array([x[:metric_len] for x in id_results]).reshape(-1))))  
    return results_freq.mean()

def calACLT(results, freq_dict, high_bar, metric_len):
    long_tail_items = list(filter(lambda x: freq_dict[x] < high_bar, list(freq_dict.keys())))
    results_cutoff = list(map(lambda x: x[:metric_len], results))
    long_tail_hit = list(map(lambda x: x in long_tail_items, np.array(results_cutoff).reshape(-1).astype(str)))
    return sum(long_tail_hit)/len(long_tail_hit)

def calPopREO(hit_results, quantiles, freq_dict, id_targets):
    groups = []
    for i in range(5):
        groups.append(list(filter(lambda x: quantiles[i] < freq_dict.get(str(id_targets[x]), 0) <= quantiles[i+1], np.arange(len(id_targets)))))
    groups_hit = [np.array(hit_results)[k].mean() for k in groups]
    print(groups_hit)
    PopREO = np.std(groups_hit)/np.mean(groups_hit)
    return PopREO

# fairness
def calJain(hit_results, rank_results):
    ndcg = (np.array(hit_results) / np.log2(np.array(rank_results)+1))
    return (np.sum(ndcg))**2/(len(ndcg)*np.sum(ndcg**2))

# gini: only taking the items that have appeared in the top-K recommendation into consideration
def cal_gini(value_list):
    gini = np.cumsum(sorted(np.array(value_list)))
    sum = gini[-1]
    x = np.array(range(len(gini))) / len(gini)
    y = gini / sum
    B = np.trapz(y, x=x)
    A = 0.5-B
    gini_coef = A / 0.5
    return gini_coef, x, y

def calcFairMetric(rec_result, metric_len):
    cov_5 = np.array([x[:metric_len] for x in rec_result]).reshape(-1)
    gini_5 = pd.value_counts(cov_5)
    # comp_len = all_len - len_5
    # comp_set = [0]*comp_len
    gini_5 = gini_5.tolist()
    # gini_5.extend(comp_set)
    gini_result, X, Y = cal_gini(gini_5)
    # print(gini_result)
    return (gini_result)

def calDPDFreq(hit_results, rank_results, cold_user, hot_user):
    ndcg = (np.array(hit_results) / np.log2(np.array(rank_results)+1))
    cold_results = np.array(ndcg[np.array(cold_user).astype(int)]).mean()
    hot_results = np.array(ndcg[np.array(hot_user).astype(int)]).mean()
    print(hot_results, cold_results)
    return abs(hot_results-cold_results)

# diversity
def calItemCoverage(id_results, metric_len, negative_samples, test_samples, cand_len=20):
    id_freq = np.array([x[:metric_len] for x in id_results]).reshape(-1)
    id_unique = set(id_freq)
    candidate_pool = set(np.array([negative_samples[x-1][1:cand_len] for x in sample]).reshape(-1)).union(set([test_samples[x-1][-1] for x in sample]))
    coverage = len(id_unique)/len(candidate_pool)
    return coverage

def mapResultsCategorySet(result, id2categories):
    rtn_cat = set()
    for item in result:
        # print(rtn_cat)
        rtn_cat = rtn_cat.union(set(id2categories.get(item, [])))
    return list(rtn_cat)

def calShannonEntropy(result, id2categories):
    all_cats = mapResultsCategorySet(result, id2categories)
    entropy = 0
    for cat in all_cats:
        p_cat = list(map(lambda x: cat in id2categories.get(x, []), result))
        p_cat = np.array(p_cat).mean()
        entropy += p_cat*np.log(p_cat)
    entropy = -entropy
    return entropy

def calMeanShannon(results, id2categories, metric_len):
    results_cutoff = list(map(lambda x: x[:metric_len], results))
    entropies = list(map(lambda x: calShannonEntropy(x, id2categories), results_cutoff))
    return np.array(entropies).mean()

# Jaccard
def calJaccard(x, y, x_cand, y_cand):
    if len(set(x_cand).intersection(set(y_cand))):
        return len(set(x).intersection(set(y)))/len(set(x_cand).intersection(set(y_cand)))
    else:
        return -1
def calMeanJaccard(results, metric_len, cand_list):
    results_cutoff = list(map(lambda x: x[:metric_len], results))
    jaccard = []
    for i in range(len(results_cutoff)):
        for j in range(i+1, len(results_cutoff)):
            tmp = calJaccard(results_cutoff[i], results_cutoff[j], cand_list[i], cand_list[j])
            if tmp != -1:
                jaccard.append(tmp)
    return sum(jaccard)/len(jaccard)


# novelty
def calSerendipity(hit_results, hit_results_pop):
    serendipity = (np.array(hit_results) - np.array(hit_results_pop)*np.array(hit_results)).mean()
    return serendipity

def calSelfInfo(results, metric_len, information):
    results_cutoff = list(map(lambda x: x[:metric_len], results))
    info_score = list(map(lambda x: information.get(x, 0), np.array(results_cutoff).astype(str).reshape(-1)))
    return np.array(info_score).mean()

def calHallucination(rec_results):
    hallucination = np.array(rec_results).reshape(-1) == 0
    return hallucination.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    args, extras = parser.parse_known_args()

    dataset = args.dataset
    results_path = args.results_path
    sota = args.sota
    model_name = args.model_name
    test_len = args.sample_num
    metric_len = args.metric_len
    save = args.save
    history_max = args.history_max
    shuffle = args.shuffle_history
    random_seed = args.random_seed
    cand_len = args.cand_len

    # with open(f'./results/{dataset}/results_gpt3.5_seed0_test_history10_multi_candidate20_shuffle_sample_1k_all.json', 'r') as file:
    #     content = file.read()
    #     prompts = json.loads(content)

    # sample user id
    with open(f'../../llm_results/{dataset}/sample.txt', 'r') as f:
        sample = eval(f.readline())
    sample = sorted(sample)
    # with open(f'results/{dataset}/sample_1k_all.txt', 'r') as f:
    #     content = f.readlines()
    # sample = [int(x[:-1]) for x in content]

    # popularity bias
    name2item_id, id2categories = read_meta(dataset)
    name_str = '\n'.join(list(name2item_id.keys()))
    # there exists ~40 items without title

    # user history
    # 1: user_id; 2-n: user_history item_id
    test_samples = read_list(f"../../data/{dataset}/sequential_data.txt")
    negative_samples = read_list(f"../../data/{dataset}/negative_samples.txt")

    if sota:
        id_results, id_targets = read_sota_results(results_path, model_name, test_samples, sample, history_max, shuffle, random_seed)
    else:
        id_results, id_targets = read_llm_results(results_path, name2item_id, name_str, test_len)

    # MostPop Baseline
    # popularity bias
    user_his_len = {x[0]: len(x[1:]) for x in test_samples}
    train_samples = [x[1:-1] for x in test_samples]
    train_freq = []
    for item in train_samples:
        train_freq += item
    train_freq = np.array(train_freq)
    train_unique = set(train_freq)
    train_id_freq_dict = dict(Counter(train_freq))
    print('train item amounts:', len(train_unique))

    pop_result, acc_result, hit_pop, cand_list = generate_mostpop(test_samples, negative_samples, sample, cand_len=cand_len) #, sample_cand_list=[eval(x['cand_list']) for x in prompts])

    # accuracy metrics
    rank = [np.argwhere(np.array(id_results[i]) == int(id_targets[i]))[0][0]+1 if len(np.argwhere(np.array(id_results[i]) == int(id_targets[i]))) else cand_len for i in range(len(id_results))]
    hit = np.array(list(map(lambda x: int(x>0 and x<=metric_len), rank)))
    # id_targets = list(map(lambda x: test_samples[x-1][-1], sample))
    # id_results = pop_result[:]
    # hit = hit_pop[:]
    # rank = acc_result[:]

    print('HR:', calHR(hit))
    print('NDCG:', calNDCG(hit, rank))
    print('MRR:', calMRR(hit, rank))

    train_freq_value = list(train_id_freq_dict.values())
    high_bar = np.quantile(train_freq_value, 0.8)
    quantiles = [np.quantile(train_freq_value, x) for x in np.arange(0, 1.2, 0.2)]
    quantiles[0] -= 1
    print('ARP:', calARP(train_id_freq_dict, id_results, metric_len))
    print('ACLT:', calACLT(id_results, train_id_freq_dict, high_bar, metric_len))
    print('PopREO:', calPopREO(hit, quantiles, train_id_freq_dict, id_targets))
    
    print('Gini:', calcFairMetric(id_results, metric_len))
    print('Jain:', calJain(hit, rank))
    user_his_len_freq = list(user_his_len.values())
    user_bar = np.quantile(user_his_len_freq, 0.5)
    print(user_bar)
    cold_user = list(filter(lambda x: user_his_len[str(sample[x])] <= user_bar, np.arange(0, test_len)))
    hot_user = list(filter(lambda x: user_his_len[str(sample[x])] > user_bar, np.arange(0, test_len)))
    print('DPD:', calDPDFreq(hit, rank, cold_user, hot_user))

    print('ItemCoverage:', calItemCoverage(id_results, metric_len, negative_samples, test_samples, cand_len=cand_len))
    print('Shannon:', calMeanShannon(id_results, id2categories, metric_len))
    print('Jaccard:', calMeanJaccard(id_results, metric_len, cand_list))

    print('Serendipity:', calSerendipity(hit, hit_pop))
    information = {k: np.log2(len(negative_samples)/v) for k, v in train_id_freq_dict.items()}
    print('SelfInformation:', calSelfInfo(id_results, metric_len, information))

    # hallucination
    print('Hallucination:', calHallucination(id_results))

    final_result = [calHR(hit), calNDCG(hit, rank), calACLT(id_results, train_id_freq_dict, high_bar, metric_len),
                     calARP(train_id_freq_dict, id_results, metric_len), calPopREO(hit, quantiles, train_id_freq_dict, id_targets),
                     calSerendipity(hit, hit_pop), calSelfInfo(id_results, metric_len, information), calcFairMetric(id_results, metric_len), 
                     calDPDFreq(hit, rank, cold_user, hot_user), calJain(hit, rank), calItemCoverage(id_results, metric_len, negative_samples, test_samples, cand_len=cand_len),
                     calMeanShannon(id_results, id2categories, metric_len), calMeanJaccard(id_results, metric_len, cand_list), calHallucination(id_results)]

    print('final_results ', final_result)
    if save:
        if sota:
            pd.DataFrame([final_result]).T.to_csv('../../sota_results/'+dataset+'/rec-'+model_name+'_metric'+str(metric_len)+'_history'+str(history_max)+'_shuffle'+str(shuffle)+'_seed'+str(random_seed)+'.csv', index=False, header=None)
        else:
            pd.DataFrame([final_result]).T.to_csv(results_path+'_'+str(metric_len)+'.csv', index=False, header=None)