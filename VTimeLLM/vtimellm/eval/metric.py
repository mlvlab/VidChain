import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dvc_eval import eval_dvc, eval_soda

import json
import argparse
import re
import difflib

import psutil
def set_cpu_affinity(start_idx=0,end_idx=128):
    p = psutil.Process()
    p.cpu_affinity(list(range(start_idx,end_idx)))

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

def save_metrics(metrics, path, num_logs):
    # if path does not exist, create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.2f}\n")
        
        f.write(f"Num samples: {num_logs}\n")

def merge_similar_sentences(data):
    if not data: return data
    merged_data = []
    current_sentence = data[0]["sentence"]
    current_timestamp = data[0]["timestamp"]
    for i in range(1, len(data)):
        next_sentence = data[i]["sentence"]
        next_timestamp = data[i]["timestamp"]
        if difflib.SequenceMatcher(None, current_sentence, next_sentence).ratio() > 0.98 and -1 <= next_timestamp[0] - current_timestamp[1] <= 1:
            current_timestamp = [current_timestamp[0], next_timestamp[1]]
        else:
            merged_data.append({"sentence": current_sentence, "timestamp": current_timestamp})
            current_sentence = next_sentence
            current_timestamp = next_timestamp
    merged_data.append({"sentence": current_sentence, "timestamp": current_timestamp})
    return merged_data

def captioning_metrics(all_logs, data_path, print_matrix, args):
    logs = [x for x in all_logs if x['task'] == 'captioning']
    pred = {}
    num_duplicates = 0

    for log in logs:
        id = log['video_id']
        answer = log['answer']
        pred[id] = []

        if args.reproduced:
            pattern = r'from (\d+) to (\d+)'
            
            try:
                items = answer.split(".")[:-1]

                all_items = []
                for i in items:
                    all_items.extend(i.split(', '))

                pattern = r'(\d+) to (\d+)'
                
                
                items = all_items
                sentences = [ i for i in items if re.search(pattern, i, re.IGNORECASE) is None]
                timestamps = [ i for i in items if  re.search(pattern, i, re.IGNORECASE) is not None ]

                for sen, time in zip(sentences, timestamps):
                    sen = sen.strip()[:]
                    time = time.strip()[:]
                    matches = re.search(pattern, time, re.IGNORECASE)
                    pred[id].append({
                            'timestamp': [int(matches.group(1)), int(matches.group(2))],
                            'sentence': sen,
                        })


            except Exception as e:
                print("Error", e, answer)
    
        else:
            try:
                items = answer.split(".")[:-1]

                for num in range(0, len(items) // 2, 2):
                    sen = items[num].strip()[4:]
                    time = items[len(items) // 2 + num ].strip()[4:]
                    pred[id].append({
                            'timestamp': [int(time[5:7]), int(time[-2:])],
                            'sentence': sen,
                        })

            except Exception as e:
                print("Error", e, answer)

    

        
        refined_pred = []
        for num_pred, curr_pred in enumerate(pred[id]):
            duplicate = False
            for curr_pred2 in pred[id][num_pred + 1:]:
            
                if curr_pred2 == curr_pred:
                    num_duplicates+=1
                    duplicate=True
                
            
            if not duplicate:
                refined_pred.append(curr_pred)

        pred[id] = refined_pred


    print(f"{num_duplicates} have been removed")
    print(len(pred))
    gt_js = json.load(open(data_path))
    gt_js = {k: v for k, v in gt_js.items() if k in pred.keys()}

    
    for id, items in list(pred.items()): 
        items = merge_similar_sentences(items)
        duration = gt_js[id]['duration']
        for item in items:
            item['timestamp'][0] = item['timestamp'][0] * duration / 100
            item['timestamp'][1] = (item['timestamp'][1] + 1) * duration / 100
        pred[id] = items
    
    pred_result = {'results': pred}
    metrics = eval_soda(pred_result, [gt_js], print_matrix=print_matrix)

    metrics.update(eval_dvc(pred_result, [gt_js], 
                tious=[0.3, 0.5, 0.7, 0.9], 
                distances=[],
                max_proposals_per_video=1000, 
                verbose=False, 
                no_lang_eval=False))
    print(f"Found {len(pred)} logs")
    metrics = {k: v * 100 for k, v in metrics.items() if k in ['soda_c', 'METEOR', 'CIDEr']}
    return metrics


def grounding_metrics(all_logs):
    ious = [x['info']['iou'] for x in all_logs if x['task'] == 'grounding']
    l = len(ious)
    print(f"Found {l} logs")
    if l == 0: return
    metrics = {
        "mIoU": sum(ious) / l * 100
    }
    for m in [0.3, 0.5, 0.7]:
        metrics[f"R1@{m}"] = sum(iou >= m for iou in ious) / l * 100
    return metrics

if __name__ == "__main__":
    set_cpu_affinity(start_idx=0,end_idx=128)
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/example_log.txt')
    parser.add_argument("--task", type=str, default='all', choices=['all', 'grounding', 'captioning'])
    parser.add_argument("--data_path", type=str, default='vtimellm/eval/data_example.json')
    parser.add_argument('--result_path', type=str, default='vtimellm/eval/result/result.txt')
    parser.add_argument("--reproduced", action='store_true')
    parser.add_argument("--print_matrix", action='store_true')
    args = parser.parse_args()

    logs = []
    with open(args.log_path) as f:
        for line in f:
            try:
                json_data = json.loads(line)
                logs.append(json_data)
            except Exception as e:
                print(e, line)

    if args.task in ['captioning', 'all']:
        print("====================== Captioning =====================")
        cap_metrics = captioning_metrics(logs, args.data_path, print_matrix=args.print_matrix, args=args)
        print_metrics(cap_metrics)
        save_metrics(cap_metrics, args.result_path, num_logs=len(logs))
    if args.task in ['grounding', 'all']:
        print("====================== Grounding ======================")
        grnd_metrics = grounding_metrics(logs)
        print_metrics(grnd_metrics)
        save_metrics(grnd_metrics, args.result_path,  num_logs=len(logs))

