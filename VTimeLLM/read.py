import json
import random
import numpy as np

random.seed(42)

def captioning_metrics(all_logs):
    logs = [x for x in all_logs if x['task'] == 'captioning']
    pred = {}
    num_duplicates = 0

    for log in logs:
        id = log['video_id']
        answer = log['answer']
        pred[id] = []


        pattern = r'from (\d+) to (\d+)'
        
        try:
            items = answer.split(".")[:-1]

            all_items = []
            for i in items:
                # all_items.extend(i.split(', from'))
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
            breakpoint()
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
    return pred


orig = json.load(open('./act_orig.json'))
cap = json.load(open('./act_capfirst.json'))
time = json.load(open('./act_timefirst.json'))
gt_json = json.load(open('./data/activitynet/val_2.json'))


orig_dict = {data['vid']: data for data in orig}
cap_dict = {data['vid']: data for data in cap}
time_dict = {data['vid']: data for data in time}

orig_score_dict = {data['vid']: data['f1-score'] for data in orig}
cap_score_dict = {data['vid']: data['f1-score'] for data in cap}
time_score_dict = {data['vid']: data['f1-score'] for data in time}


orig_score_sort = dict(sorted(orig_score_dict.items(), key=lambda item: item[1]))
cap_score_sort = dict(sorted(cap_score_dict.items(), key=lambda item: item[1], reverse=True))
time_score_sort = dict(sorted(time_score_dict.items(), key=lambda item: item[1], reverse=True))

num_display = 20


this_cap = list(cap_score_sort.keys())[:50]
this_time = list(time_score_sort.keys())[:50]

# breakpoint()

# for i in range(num_display):
#     print(cap_score_sort[list(cap_score_sort.keys())[i]])

# print()
# for i in range(num_display):
#     print(time_score_sort[list(time_score_sort.keys())[i]])

cap_score_int= {data['vid']: data['f1-score'] for data in cap }
time_score_int = {data['vid']: data['f1-score'] for data in time if data['vid'] in cap_dict.keys()}


video_ids = []
average = []

print("Printing relative")
num = 0
for key in cap_dict.keys():
    # if cap_score_int[key] > time_score_int[key] + 13:
    if cap_score_int[key] > time_score_int[key] + 13:
        # print(f'cap score: {cap_score_int[key]}')
        # print(f'time score: {time_score_int[key]}')
        print(key)
        # print(key, gt_json[key]['duration'])
        average.append( gt_json[key]['duration'])
        video_ids.append(key)
        num += 1
    
print(num)
print(np.mean(average))

average = []


print("Printing relative")
num = 0
for key in cap_dict.keys():
    if cap_score_int[key] + 20 < time_score_int[key]:
        # print(f'cap score: {cap_score_int[key]}')
        # print(f'time score: {time_score_int[key]}')
        # print(key, gt_json[key]['duration'])
        print(key)
        average.append( gt_json[key]['duration'])


        video_ids.append(key)
        num += 1
    
print(num)
print(np.mean(average))

average = []

ran_cap = list(cap_dict.keys())
random.shuffle(ran_cap)
num = 0
print()

for key in ran_cap[:20]:
    # print(f'cap score: {cap_score_int[key]}')
    # print(f'time score: {time_score_int[key]}')
    print(key, gt_json[key]['duration'])
    average.append( gt_json[key]['duration'])
    video_ids.append(key)
    num += 1




# random.shuffle(video_ids)
# print(len(set(video_ids)), len(video_ids))
# for vid in video_ids:
#     print(vid)
print(num)
print(np.mean(average))


for key in time_dict.keys():

    # if key not in [['v_WCChCrg9eZU,', 'v_Y5uVICaJU-0', 'v_MaJlWFemO68', 'v_uF9othvTXn8', 'v_VRwI8Iydb_o', 'v_Zr8cz8QrBp4', 'v_n1iu-AlcS-Q', 'v_KbbEbeCJTJg', 'v_xcBJP14YBvg', 'v_4HxmQpkryjA', 'v_pMDFkrK0KRc', 'v_1VSqWp5DZiU', 'v_AeefhelpxGA', 'v_D_y9uXMbImA', 'v__roK9m9UOvM', 'v_QjMNQxu3Zf8', 'v_ISHKwbnOzXY', 'v_NDyc4PZE954', 'v_Ue90f5r-2Qw']]:
    #     continue

    if key not in this_cap:
        continue

    # if key not in ["v_-sd2XAFkeC0",
    #         "v_BodF651KcIg",
    #         "v_D7Oc3SLX0wo",
    #         "v_RrEJ2-TfWCI",
    #         "v_rNb4Jz_t9F4",
    #         "v_Qu3_80O0j5w",
    #         "v_0EdDWY0Zuqw",
    #         "v_mNM01g9wLy4",
    #         "v_ox6cIfguQ00",
    #         "v_FcfoTk3UK5g",
    #         "v_EVtM8DKW4bc",
    #         "v_AeOUzM7nl5w",
    #         "v_Nq3b9OReeEI",
    #         "v_A7ER02-zr54",
    #         "v__1vYKA7mNLI",
    #         "v_dQR6VEemP24",
    #         "v_2g9GrshWQrU",
    #         "v_aoIGBV31OT4",
    #         "v_1RVu0qNtWCc",
    #         "v_QOuNt8YH3Rk",
    #         "v_H0l29-F7Edg",
    #         "v_qI1ZayfiGHI",]:
    #     continue


    print(f'video ID: {key}')
    try:
        curr_orig = orig_dict[key]
        print("="*150)
        print("Original")
        print("="*150)
        print(f'precision: {curr_orig["precision"]}')
        print(f'recall: {curr_orig["recall"]}')
        print(f'f-1: {curr_orig["f1-score"]}\n')

        for (p_s, p_t) in zip(curr_orig['pred_sentences'], curr_orig['pred_timestamps']):
            print(f'Time: {p_t[0]:.1f}, {p_t[1]:.1f} {p_s}')
        print("="*150)


    except:
        print("key not in orig_dict")

    print()
    

    try:
        curr_cap = cap_dict[key]
        print("="*150)
        print("Cap-first")
        print("="*150)
        print(f'precision: {curr_cap["precision"]}')
        print(f'recall: {curr_cap["recall"]}')
        print(f'f-1: {curr_cap["f1-score"]}\n')
        print(gt_json[key]['duration'])


        for (p_s, p_t) in zip(curr_cap['pred_sentences'], curr_cap['pred_timestamps']):
            print(f'Time: {p_t[0]:.1f}, {p_t[1]:.1f} {p_s}')

        print("="*150) 
        
    except:
        print("key not in cap_dict")
        
         
    print()

    try:
        curr_time = time_dict[key]
        print("="*150)
        print("Time-first")
        print("="*150)

        print(f'precision: {curr_time["precision"]}')
        print(f'recall: {curr_time["recall"]}')
        print(f'f-1: {curr_time["f1-score"]}\n')
        print(gt_json[key]['duration'])
        
        for (p_s, p_t) in zip(curr_time['pred_sentences'], curr_time['pred_timestamps']):
            print(f'Time: {p_t[0]:.1f}, {p_t[1]:.1f} {p_s}')

        print("="*150)

        print()
        print("*"*50)
        print("GOLD")
        for (p_s, p_t) in zip(curr_time['gold_sentences'], curr_time['gold_timestamps']):
            print(f'Time: {p_t[0]:.1f}, {p_t[1]:.1f} {p_s}')
        print("*"*50)


    except:
        print("key not in time_dict")
    

    print()
    print()


    breakpoint()

breakpoint()


# for og, cp, te in zip(orig, cap, time):
#                 print("=" * 100)
#             print(f'Video ID: {vid}')
#             print("======= IOU Matrix =======")
#             print(_iou * 100)
#             print("\n======= SCORE Matrix =======")
#             print(scores * 100)
#             print("\n======= COST Matrix =======")
#             print(_iou * scores * 100)
#             print()
#             print(f'Precision: {p * 100}\nRecall: {r * 100}\nF1-score: {_f[i][0] * 100}')
#             print("Prediction")

#             for (p_s, p_t) in zip(pred['sentences'], pred['timestamps']):
#                 print(f'Time: {p_t[0]:.1f}, {p_t[1]:.1f} {p_s}')
#             # print(pred['sentences'])
#             # print(pred['timestamps'])
#             print()
#             print("Ground-truth")
#             for (p_s, p_t) in zip(gold['sentences'], gold['timestamps']):
#                 print(f'Time: {p_t[0]:.1f}, {p_t[1]:.1f} {p_s}')