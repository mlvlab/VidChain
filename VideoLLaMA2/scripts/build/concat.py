import json
import random

#########################################################################################
#===== ActivitiyNet Config =====#
dpo_folder_path = f"./outputs/mdpo-dataset/videollama2/generated-samples-act/dpo"
data_path = f"./data/activitynet"


#===== YouCook2 Config =====#
# dpo_folder_path = f"./outputs/mdpo-dataset/videollama2/generated-samples-youcook/dpo"
# data_path = f"./data/YouCook2"
#########################################################################################


DPO_DATASETS = [ 
    f"{dpo_folder_path}/1_mdpo_timefirst_t2.json",  # Time is generated
    f"{dpo_folder_path}/2_mdpo_capfirst_t2.json",  # Caption is generated
    f"{dpo_folder_path}/3_mdpo_timefirst.json",  # Time is GT, Caption is generated
    f"{dpo_folder_path}/4_mdpo_capfirst.json",  # Caption is GT, Time is generated
]

def distribute_samples(data, target_per_index):
    # Initialize the result list
    result = [0] * len(data)
    
    # First pass: allocate up to 5000 samples per index
    for i in range(len(data)):
        result[i] = min(data[i], target_per_index)
    
    # Calculate remaining samples needed to reach 20000
    remaining_samples = target_per_index * len(data) - sum(result)
    
    # Distribute remaining samples from the largest available index
    while remaining_samples > 0:
        # Find the index with the maximum available samples
        max_index = max(range(len(data)), key=lambda i: data[i] - result[i])
        available = data[max_index] - result[max_index]
        
        if available > 0:
            take = min(available, remaining_samples)
            result[max_index] += take
            remaining_samples -= take
        else:
            break
    return result

data_c = [json.load(open(path)) for path in DPO_DATASETS]
num_samples = 5000 # Then total will be 20000 (5000 from each file)

random.seed(42)
_len = [len(d['vid']) for d in data_c]
num4each = distribute_samples(_len, num_samples)

print("Current Path: ", dpo_folder_path)
print("Length of each data: ", _len)
print("Sum of length: ", sum(_len))
print("Num of samples for each: ", num4each)
print("Sum of num4each: ", sum(num4each))

final_dict = {k:[] for k in ['prompt', 'chosen', 'rejected', 'vid', 'preference', 'preference_accept', 'preference_reject']}

for curr_i, curr_data in enumerate(data_c):
    select = [i for i in range(len(curr_data['vid']))]
    random.shuffle(select) # random select
    select = select[:num4each[curr_i]]
    
    for k in ['prompt', 'chosen', 'rejected', 'vid', 'preference', 'preference_accept', 'preference_reject']:
        selected = [d for i,d in enumerate(curr_data[k]) if i in select ]
        final_dict[k].extend(selected)

print([len(v) for i,v in final_dict.items()])
with open( f'{data_path}/mdpo-train.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
