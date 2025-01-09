import json
import re
import random
from num2words import num2words
import numpy as np
import os

random.seed(42)

densecap_query = "<video>\nCould you please detail the events that took place during different time segments in the video?" # <CAPTION>, from <START to <END> etc or From <START> to <END>, <CAPTION> etc
densecap_query_wo_vid = "Could you please detail the events that took place during different time segments in the video?" # <CAPTION>, from <START to <END> etc or From <START> to <END>, <CAPTION> etc
vidcap_query = "<video>\nCan you explain what happened in the video?" # <CAPTION1> <CAPTION2> <CAPTION3> <CAPTION4>
vidseg_query = "<video>\nCan you breakdown the video into different time segments?" # First event from <START> to <END>, second event from <START> to <END> ...


vid_grounding_query = "During which frames can we see <CAPTION> in the video?" # From <START> to <END>
clip_captioning_query = "Can you describe what occurred from <START> to <END> in the video?" # <CAPTION>
time_segments_query = "How many of time segments can this video breakdown into?"


def convert(duration, x):
    x = x / duration * 100
    x = str(min(round(x), 99))

    if len(x) == 1:
        x = "0" + x
    return x

def capFirst(cur_cap, duration, task2=False, eval_method = "", args=None):
    conv = []
    # ====================================================================================================================== #
    conv.append({ 'from': 'human', 'value': f'<video>\n{time_segments_query}', })
    conv.append({ 'from': 'gpt', 'value': f'{len(cur_cap["pred_sentences"])} time segments'})
    # ====================================================================================================================== #

    # P(C|V) <CAPTION>. <CAPTION>. <CAPTION>. 
    
    if task2: 
         # ====================================================================================================================== #
        conv.append({ 'from': 'human', 'value': vidcap_query.replace("<video>\n", "")})
        conv.append({ 'from': 'gpt', 'value': cur_cap['original'].strip()})
        # ====================================================================================================================== #
        return conv
    else:
        
        if args.youcook2:
            second_output = ". ".join([i[0].capitalize() for i in cur_cap['gold_sentences']])
            # ====================================================================================================================== #
            conv.append({ 'from': 'human', 'value': vidcap_query.replace("<video>\n", "")})
            if second_output.strip()[-1] != ".":
                second_output = second_output.strip() + "."
            conv.append({ 'from': 'gpt', 'value': second_output.strip()})
            
            
            all_sentences = cur_cap['original'].split('.')[:-1]
            this_timestamps = []
            for sen in all_sentences:
                pattern = r'from (\d+) to (\d+)'
                
                matches = re.search(pattern, sen, re.IGNORECASE)
                this_timestamps.append([matches.group(1), matches.group(2)])
            
            conv.append({ 'from': 'human', 'value': "What are the time segments for each event?", })    
            conv_value = []
            for num_i, num in enumerate(this_timestamps):
                text = num2words(num_i + 1, to='ordinal')
                conv_value.append(f'{text.capitalize()} event, from {num[0]} to {num[1]}.')

            conv_value = " ".join(conv_value)
            conv.append({'from':'gpt', 'value': conv_value})
            return conv
            
        else:
            all_sentences = cur_cap['original'].split('.')[:-1]
            all_sentences = [s+'.' for s in all_sentences]
            
            this_sentences = [sen.strip() for sen in all_sentences if re.search(r'from (\d+) to (\d+)', sen, re.IGNORECASE) is  None]
            this_timestamps = [sen.strip() for sen in all_sentences if re.search(r'from (\d+) to (\d+)', sen, re.IGNORECASE) is not None]
                    
            second_output = " ".join([i.capitalize() for i in this_sentences])
            # ====================================================================================================================== #
            conv.append({ 'from': 'human', 'value': vidcap_query.replace("<video>\n", "")})
            
            
            if second_output.strip()[-1] != ".":
                second_output = second_output.strip() + "."
            conv.append({ 'from': 'gpt', 'value': second_output.strip()})
        # ====================================================================================================================== #
        
        # P(T| C, V) Frist event , from <START> to <END>. Second event, from <START> to <END> 
        # ====================================================================================================================== #
        conv.append({ 'from': 'human', 'value': "What are the time segments for each event?", })
        third_output = " ".join(this_timestamps)
        if third_output.strip()[-1] != ".":
            third_output = third_output.strip() + "."
        conv.append({'from':'gpt', 'value': third_output})
        # ====================================================================================================================== #
        return conv


def capFirstGT(cur_cap, duration, task2=False):
    conv = []
    # ====================================================================================================================== #
    conv.append({ 'from': 'human', 'value': f'<video>\n{time_segments_query}', })
    conv.append({ 'from': 'gpt', 'value': f'{len(cur_cap["gold_sentences"])} time segments'})
    # ====================================================================================================================== #

    # P(C|V) <CAPTION>. <CAPTION>. <CAPTION>. 
    second_output = ". ".join([i[0].capitalize() for i in cur_cap['gold_sentences']])
    # ====================================================================================================================== #
    conv.append({ 'from': 'human', 'value': vidcap_query.replace("<video>\n", "")})
    if second_output.strip()[-1] != ".":
        second_output = second_output.strip() + "."
    conv.append({ 'from': 'gpt', 'value': second_output.strip()})
    # ====================================================================================================================== #
    if task2: return conv
    
    gt_timestamps = [ (convert(duration, t[0]), convert(duration, t[1])) for t in cur_cap['gold_timestamps']]
    

    # P(T| C, V) Frist event , from <START> to <END>. Second event, from <START> to <END> 
    # ====================================================================================================================== #
    conv.append({ 'from': 'human', 'value': "What are the time segments for each event?", })
    conv_value = []
    for num_i, num in enumerate(gt_timestamps):
        text = num2words(num_i + 1, to='ordinal')
        conv_value.append(f'{text.capitalize()} event, from {num[0]} to {num[1]}.')

    conv_value = " ".join(conv_value)
    conv.append({'from':'gpt', 'value': conv_value})
    # ====================================================================================================================== #
    
    return conv


def timeFirst(cur_cap, task2=False,  eval_method = "", args=None):
    conv = []
    # ====================================================================================================================== #
    conv.append({ 'from': 'human', 'value': f'<video>\n{time_segments_query}', })
    conv.append({ 'from': 'gpt', 'value': f'{len(cur_cap["pred_sentences"])} time segments'})
    # ====================================================================================================================== #

    # P(C|V) <CAPTION>. <CAPTION>. <CAPTION>. 
    this_sentences = cur_cap['original'].split('.')[:-1]
    this_sentences = [s+'.' for s in this_sentences]
    
    pattern = r'from (\d+) to (\d+)'
    this_sentences = [sen for sen in this_sentences if re.search(pattern, sen, re.IGNORECASE) is not None]
    
    
    this_timestamps = []
    for sen in this_sentences:
        pattern = r'from (\d+) to (\d+)'
        
        matches = re.search(pattern, sen, re.IGNORECASE)
        this_timestamps.append([matches.group(1), matches.group(2)])

        
    conv_value = []
    for num in range(len(this_timestamps)):
        text = num2words(num + 1, to='ordinal')
        conv_value.append(f'{text.capitalize()} event, from {this_timestamps[num][0]} to {this_timestamps[num][1]}.')
    
    conv_value = " ".join(conv_value)
    conv.append({ 'from': 'human', 'value': "Can you breakdown the video into different time segments?", })
    conv.append({'from':'gpt', 'value': conv_value})
    if task2: return conv
    
    conv.append({ 'from': 'human', 'value': densecap_query_wo_vid, }) # without video 
    if cur_cap["original"].strip()[-1] != ".":
        cur_cap["original"] = cur_cap["original"].strip() + "."
    conv.append({ 'from': 'gpt', 'value': f'{cur_cap["original"]}'})
    return conv
         
def TimeFirstGT(cur_cap, duration, task2=False):
    conv = []
    # ====================================================================================================================== #
    conv.append({ 'from': 'human', 'value': f'<video>\n{time_segments_query}', })
    conv.append({ 'from': 'gpt', 'value': f'{len(cur_cap["gold_sentences"])} time segments'})
    # ====================================================================================================================== #

    # P(C|V) <CAPTION>. <CAPTION>. <CAPTION>. 
    
    gt_timestamps = [ (convert(duration, t[0]), convert(duration, t[1])) for t in cur_cap['gold_timestamps']]
    
    conv.append({ 'from': 'human', 'value': "Can you breakdown the video into different time segments?", })
    conv_value = []
    for num_i, num in enumerate(gt_timestamps):
        text = num2words(num_i + 1, to='ordinal')
        conv_value.append(f'{text.capitalize()} event, from {num[0]} to {num[1]}.')

    conv_value = " ".join(conv_value)
    conv.append({'from':'gpt', 'value': conv_value})
    if task2: return conv
    
    conv.append({ 'from': 'human', 'value': densecap_query_wo_vid, }) # without video 
    sentences = [i[0] for i in cur_cap['gold_sentences']]
    dense_cap_value = []

    for num, sen in enumerate(sentences):
        time = gt_timestamps[num]
        sen = sen.strip().replace(".", "")

        sen = sen[0].lower() + sen[1:]
        dense_cap_value.append(f'From {time[0]} to {time[1]}, {sen}.')
    
    dense_cap_value = " ".join(dense_cap_value)
    conv.append({'from':'gpt', 'value': dense_cap_value})
    return conv           


def create_json(cap_first, time_first, cap_first_t2, time_first_t2, gt_json, args):
    print(f"Current Num Sample Sets of CapFirst: {len(cap_first)}")
    print(f"Current Num Sample Sets of TimeFirst: {len(time_first)}")
    print(f"Current Num Sample Sets of CapFirst T2: {len(cap_first_t2)}")
    print(f"Current Num Sample Sets of TimeFirst T2: {len(time_first_t2)}")

    vid = []
    prompt = []
    chosen = []
    rejected = []
    preference = []
    preference_accept = []
    preference_reject = []
    
    soda_diff_cap, soda_diff_t2_cap, soda_diff_time, soda_diff_t2_time = [], [], [], []
    
    for key, data in gt_json.items():
        sentences = data['sentences']
        sentences = [sen.strip() for sen in sentences]
        
        # dict_keys(['vid', 'precision', 'recall', 'preference', 'pred_sentences', 'gold_sentences', 'pred_timestamps', 'gold_timestamps', 'original'])
        conv_list_cap, conv_list_time, gt_conv_cap, gt_conv_time = [], [], [], []
        soda_score_cap, soda_score_time = [], []

        conv_list_cap_t2, conv_list_time_t2, gt_conv_cap_t2, gt_conv_time_t2 = [], [], [], []
        score_cap_t2, score_time_t2 = [], []
        
        # Only consider two pairs
        for cur_i in range(args.num_samples):
            
            if not args.task2:
                cap_data = cap_first[cur_i]
                time_data = time_first[cur_i]
            
                if key in cap_data.keys():
                    cur_cap = cap_data[key]
                    
                    try:
                        conv = capFirst(cur_cap, duration=data['duration'], args=args)
                    except:
                        conv = ""
                        print(f"{key} continued in TimeFirst")
                        continue
                
                    soda_score_cap.append(cur_cap['preference'])
                    conv_list_cap.append(conv)
                    
                    conv_gt = capFirstGT(cur_cap, data['duration'])
                    gt_conv_cap.append(conv_gt)
                    
                
                if key in time_data.keys():
                    cur_time = time_data[key]
                    
                    try:
                        conv = timeFirst(cur_time, args=args)
                    except:
                        print(f"{key} continued in TimeFirst")
                        continue
                    soda_score_time.append(cur_time['preference'])  
                    conv_list_time.append(conv)
                    
                    conv_gt = TimeFirstGT(cur_time, data['duration'])
                    gt_conv_time.append(conv_gt)
                
            # If task 2
            if args.task2 and cap_first_t2!=[] and time_first_t2 != []:
                cap_data = cap_first_t2[cur_i]
                time_data = time_first_t2[cur_i]
                
                if key in cap_data.keys():
                    cur_cap = cap_data[key]
                    conv = capFirst(cur_cap, duration=data['duration'], task2=True)
                    score_cap_t2.append(cur_cap['preference'])
                    conv_list_cap_t2.append(conv)
                    
                    conv_gt = capFirstGT(cur_cap, data['duration'], task2=True)
                    gt_conv_cap_t2.append(conv_gt)
                    
                    
                if key in time_data.keys():
                    cur_time = time_data[key]
                    conv = timeFirst(cur_time, task2=True)
                    score_time_t2.append(cur_time['preference'])  
                    conv_list_time_t2.append(conv)
                    
                    conv_gt = TimeFirstGT(cur_time, data['duration'], task2=True)
                    gt_conv_time_t2.append(conv_gt)


        # Give different threshold for the outputs 
        eval_threshold = args.eval_threshold
        
        num2pairs = {2: [(0, 1)], 3:[(0, 1), (1, 2), (0, 2)]} # possible combination
        for this_i, this_nxt in num2pairs[args.num_samples]:
            
            if not args.task2:
                # CAP FIRST
                if args.with_CAPFirst:
                    
                    if conv_list_cap != [] and len(conv_list_cap) > this_nxt:
                        first_cap = conv_list_cap[this_i]
                        second_cap = conv_list_cap[this_nxt]
                        

                        if soda_score_cap[this_i] >= soda_score_cap[this_nxt] and (soda_score_cap[this_i] - soda_score_cap[this_nxt] >= eval_threshold):
                            soda_diff_cap.append(soda_score_cap[this_i] - soda_score_cap[this_nxt])
                            prompt.append(first_cap[0])
                            chosen.append(first_cap[1:])
                            rejected.append(second_cap[1:])
                            
                            # Meta-data preference 
                            preference.append(soda_score_cap[this_i] - soda_score_cap[this_nxt])
                            preference_accept.append( soda_score_cap[this_i])
                            preference_reject.append( soda_score_cap[this_nxt])
                            vid.append(key) 
                            
                        elif soda_score_cap[this_i] <= soda_score_cap[this_nxt] and ((soda_score_cap[this_nxt] - soda_score_cap[this_i]) >= eval_threshold):
                            soda_diff_cap.append(soda_score_cap[this_nxt] - soda_score_cap[this_i])
                            
                            prompt.append(first_cap[0])
                            chosen.append(second_cap[1:])
                            rejected.append(first_cap[1:])
                            
                             # Meta-data preference 
                            preference.append(soda_score_cap[this_nxt] - soda_score_cap[this_i])
                            preference_accept.append( soda_score_cap[this_nxt])
                            preference_reject.append( soda_score_cap[this_i])
                            vid.append(key)
                        else:
                            continue
                        
                    

                if args.with_TIMEFirst:
                    # TIME FIRST
                    if conv_list_time != [] and len(conv_list_time) > this_nxt:
                        first_time = conv_list_time[this_i]
                        second_time = conv_list_time[this_nxt]
                    
                        
                        if soda_score_time[this_i] >= soda_score_time[this_nxt] and (soda_score_time[this_i] - soda_score_time[this_nxt] >= eval_threshold):
                            soda_diff_time.append(soda_score_time[this_i] - soda_score_time[this_nxt])
                            
                            prompt.append(first_time[0])
                            chosen.append(first_time[1:])
                            rejected.append(second_time[1:])
                            
                            # Meta-data preference 
                            preference.append(soda_score_time[this_i] - soda_score_time[this_nxt])
                            preference_accept.append( soda_score_time[this_i])
                            preference_reject.append( soda_score_time[this_nxt])
                            vid.append(key)
                            
                        elif soda_score_time[this_i] <= soda_score_time[this_nxt] and (soda_score_time[this_nxt] - soda_score_time[this_i] >= eval_threshold):
                            soda_diff_time.append(soda_score_time[this_nxt] - soda_score_time[this_i])
                            
                            prompt.append(first_time[0])
                            chosen.append(second_time[1:])
                            rejected.append(first_time[1:])
                            
                            # Meta-data preference 
                            preference.append(soda_score_time[this_nxt] - soda_score_time[this_i])
                            preference_accept.append( soda_score_time[this_nxt])
                            preference_reject.append( soda_score_time[this_i])
                            vid.append(key)
                        else:
                            continue
                        
                
            else:
                if args.with_TIMEFirst:
                    if conv_list_time_t2 != [] and len(conv_list_time_t2) > this_nxt:
                        first_time = conv_list_time_t2[this_i]
                        second_time = conv_list_time_t2[this_nxt]
                            
                        if score_time_t2[this_i] >= score_time_t2[this_nxt] and (score_time_t2[this_i] - score_time_t2[this_nxt] >= eval_threshold):
                            soda_diff_t2_time.append(score_time_t2[this_i] - score_time_t2[this_nxt])
                            
                            prompt.append(first_time[0])
                            chosen.append(first_time[1:])
                            rejected.append(second_time[1:])
                            
                            # Meta-data preference 
                            preference.append(score_time_t2[this_i] - score_time_t2[this_nxt])
                            preference_accept.append( score_time_t2[this_i])
                            preference_reject.append( score_time_t2[this_nxt])
                            vid.append(key)
                            
                        elif score_time_t2[this_i] <= score_time_t2[this_nxt] and (score_time_t2[this_nxt] - score_time_t2[this_i] >= eval_threshold):
                            soda_diff_t2_time.append(score_time_t2[this_nxt] - score_time_t2[this_i])
                            
                            prompt.append(first_time[0])
                            chosen.append(second_time[1:])
                            rejected.append(first_time[1:])
                            
                            # Meta-data preference 
                            preference.append(score_time_t2[this_nxt] - score_time_t2[this_i])
                            preference_accept.append( score_time_t2[this_nxt])
                            preference_reject.append( score_time_t2[this_i])
                            vid.append(key)
                        else:
                            continue
                        
                        
                
                if args.with_CAPFirst:
                    if conv_list_cap_t2 != [] and len(conv_list_cap_t2) > this_nxt:
                        first_cap = conv_list_cap_t2[this_i]
                        second_cap = conv_list_cap_t2[this_nxt]
                        
                        # CAP FIRST T2
                        if score_cap_t2[this_i] > score_cap_t2[this_nxt] and (score_cap_t2[this_i] - score_cap_t2[this_nxt]) >= eval_threshold:
                            soda_diff_t2_cap.append(score_cap_t2[this_i] - score_cap_t2[this_nxt])
                            
                            prompt.append(first_cap[0])
                            chosen.append(first_cap[1:])
                            rejected.append(second_cap[1:])
                            
                            # Meta-data preference 
                            preference.append(score_cap_t2[this_i] - score_cap_t2[this_nxt])
                            preference_accept.append( score_cap_t2[this_i])
                            preference_reject.append( score_cap_t2[this_nxt])
                            vid.append(key)
                            
                        elif score_cap_t2[this_i] < score_cap_t2[this_nxt] and ((score_cap_t2[this_nxt] - score_cap_t2[this_i]) >= eval_threshold):
                            soda_diff_t2_cap.append(score_cap_t2[this_nxt] - score_cap_t2[this_i])
                            
                            prompt.append(first_cap[0])
                            chosen.append(second_cap[1:])
                            rejected.append(first_cap[1:])
                            
                            # Meta-data preference 
                            preference.append(score_cap_t2[this_nxt] - score_cap_t2[this_i])
                            preference_accept.append( score_cap_t2[this_nxt])
                            preference_reject.append( score_cap_t2[this_i])
                            vid.append(key)
                        else:
                            continue
                        

    final_dict = {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'vid': vid,
        'preference': preference,
        'preference_accept': preference_accept,
        'preference_reject': preference_reject,
        
    }
    
    
    ###################################################################################################################################
    # Make sure the num data is consistent
    num_total = []
    for i in final_dict.keys():
        num_total.append(len(final_dict[i]))
    
    assert len(set(num_total)) == 1, f"The number of samples are NOT consistent: {num_total}"
    print(f"The number of samples are consistent with {(num_total[0])}")
    
    def prettyPrint(text, data1, max=True):
        if max and len(data1)!= 0:
            print(f'{text}: {np.max(data1)}, {len(data1)}')
        elif not max and len(data1)!= 0:
            print(f'{text}: {np.min(data1)}, {len(data1)}')
    
    
    prettyPrint("SODA CAP DIFF MAX", soda_diff_cap, max=True)
    prettyPrint("SODA CAP DIFF MIN", soda_diff_cap, max=False)
    
    prettyPrint("SODA TIME DIFF MAX", soda_diff_time, max=True)
    prettyPrint("SODA TIME DIFF MIN", soda_diff_time, max=False)
    
    prettyPrint("SODA CAP DIFF MEAN", soda_diff_cap, max=False)
    prettyPrint("SODA TIME DIFF MEAN", soda_diff_time, max=False)
    
    
    prettyPrint("SODA CAP DIFF T2 MAX", soda_diff_t2_cap, max=True)
    prettyPrint("SODA CAP DIFF T2 MIN", soda_diff_t2_cap, max=False)
    
    prettyPrint("SODA TIME DIFF T2 MAX", soda_diff_t2_time, max=True)
    prettyPrint("SODA TIME DIFF T2 MIN", soda_diff_t2_time, max=False)
    
    prettyPrint("SODA CAP DIFF T2 MEAN", soda_diff_t2_cap, max=False)
    prettyPrint("SODA TIME DIFF T2 MEAN", soda_diff_t2_time, max=False)
     ####################################################################################################################################

    
    # Save config (parameters)
    args_dict = vars(args)
    final_dict['config'] = args_dict
    
    if args.overwrite_save_directory == "":
        print("="* 150)
        print(f'Going to save to.... {args.output_file}/{args.save_file_initial}.json')
        print("="* 150)
        
        if not os.path.exists(args.output_file):
            os.makedirs(args.output_file, exist_ok=True)
        
        with open( f'{args.output_file}/{args.save_file_initial}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
        return True

    else:
        print("="* 150)
        print(f'Going to save to.... {args.overwrite_save_directory}/{args.save_file_initial}.json')
        print("="* 150)
        
        if not os.path.exists(args.overwrite_save_directory):
            os.makedirs(args.overwrite_save_directory, exist_ok=True)
        
        with open( f'{args.overwrite_save_directory}/{args.save_file_initial}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
        return True