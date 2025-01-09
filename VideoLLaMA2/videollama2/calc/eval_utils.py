import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dvc_eval import eval_dvc, eval_soda

import json
import re
import time as timer

def captioning_metrics(all_logs, data_path, print_matrix, args):
    gt_js = json.load(open(data_path))
    logs = all_logs

    all_pred = [{} for i in range(args.num_samples)]
    num_duplicates = 0
    
    eval_method = 'cap-as-GT'
    if (args.mode == 'dvc-capfirst' and args.task2) or (args.mode == 'dvc-timefirst' and not args.task2):
        # requires timestamp as GT else caption will be provided as GT
        eval_method = 'time-as-GT'
    
    print("-"* 100)
    print(f"Current mode: {args.mode} Task2: {args.task2}, EVAL MODE: {eval_method}")
    print("-"* 100)

    for log in logs:
        id = log['video_id']
        del log['answer']
        
        for curr_num in range(args.num_samples):
            answer = log[str(curr_num)]['sentence']
            all_pred[curr_num][id] = []
            pattern = r'(\d+) to (\d+)'
            
            try:
                items = answer.split(".")
                if items[-1] == '.': items = items[:-1]
                
                # timestamp_flag: Indicate if the timestamp is also present
                try:
                    timestamp_flag = True if re.search(r'(\d+) to (\d+)', items[0], re.IGNORECASE) is not None else False 
                except Exception as e:
                    print("Error", e)
                    continue
                
                if timestamp_flag: # Split by ", "; Assume From x to y, blahblahblah. 
                    all_items = []
                    for it in items:
                        pattern = r",\s+"
                        all_items.extend(re.split(pattern, it, maxsplit=1))
                    items = all_items
                                
                if eval_method == 'time-as-GT':
                    timestamps = gt_js[id]['timestamps'] # GT timestamp
                    sentences = [ i for i in items if re.search(r'(\d+) to (\d+)', i, re.IGNORECASE) is None] # Generated sentences
                    
                    for sen, time in zip(sentences, timestamps):
                        sen = sen.strip()[:]
                        all_pred[curr_num][id].append({
                                'timestamp': [time[0],time[1]], # GT
                                'sentence': sen,
                                'original': answer,
                            })
                    
                else:
                    # 'cap-as-GT' : Caption as GT
                    sentences = gt_js[id]['sentences'] # GT caption
                    timestamps = [ i for i in items if  re.search(r'(\d+) to (\d+)', i, re.IGNORECASE) is not None ] # Generated timestamp
                    
                    for sen, time in zip(sentences, timestamps):
                        sen = sen.strip()[:]
                        time = time.strip()[:]
                        matches = re.search(r'(\d+) to (\d+)', time, re.IGNORECASE)
                        all_pred[curr_num][id].append({
                                'timestamp': [int(matches.group(1)), int(matches.group(2))],
                                'sentence': sen, # GT
                                'original': answer,
                            })
            except Exception as e:
                print(f'Error: {e}, Answer: {answer}')
                continue
            
            # Remove exact same predictions (duplicates)
            refined_pred = []
            for num_pred, curr_pred in enumerate(all_pred[curr_num][id]):
                duplicate = False
                for curr_pred2 in all_pred[curr_num][id][num_pred + 1:]:
                
                    if curr_pred2 == curr_pred:
                        num_duplicates+=1
                        duplicate=True
                
                if not duplicate:
                    refined_pred.append(curr_pred)

            all_pred[curr_num][id] = refined_pred
    
    print(f"{num_duplicates} have been removed")
    print("Num samples for each task: ", len(all_pred))
    print("Each Length: ", [len(i) for i in all_pred])

    gt_js = {k: v for k, v in gt_js.items() if k in all_pred[0].keys()}    
    for num in range(args.num_samples):
        for id, items in list(all_pred[num].items()): 
            duration = gt_js[id]['duration']

            if eval_method == 'cap-as-GT':
                # Convert timestamp scale to GT scale
                for item in items:
                    item['timestamp'][0] = item['timestamp'][0] * duration / 100
                    item['timestamp'][1] = (item['timestamp'][1] + 1) * duration / 100
                all_pred[num][id] = items
        pred_result = {'results': all_pred[num]}

        final = eval_soda(pred_result, [gt_js], print_matrix=True, args=args)
        print(f"Current list length: {len(final)}")
        
        final = {thisData['vid']: thisData for thisData in final}
        for key in final.keys():
            orig_data = all_pred[num][key]
            final[key].update({
                'original': orig_data[0]['original'],
            })        
        
        if args.vtimellm:
            intermediate_path = f'{args.overwrite_save_directory}/json_dump'
        else:
            intermediate_path = f'{args.output_file}/json_dump'
        if not os.path.exists(intermediate_path):
            os.makedirs(intermediate_path)
        path = f'{intermediate_path}/{args.save_file}'
        
        if args.task2:
            with open( f'{path}_t2_{num}.json', 'w') as f: f.write(json.dumps(final, indent=2))
        else:
            with open( f'{path}_{num}.json', 'w') as f: f.write(json.dumps(final, indent=2))
            
    return True



    