import os, argparse, json, warnings, re, sys
from tqdm import tqdm

sys.path.append('./')
from videollama2.calc.eval_utils import captioning_metrics
from videollama2.calc.conv import create_json

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# Split the data into chunks for each GPU
def split2chunk(gt_questions, total_gpu, num_gpu):
    total_data = len(gt_questions)
    each_gpu = total_data // total_gpu
    gt_keys = list(gt_questions.keys())
    
    print("=" * 90)
    if num_gpu == total_gpu - 1:
        print("Inside left overs ")
        curr_gt_keys = gt_keys[num_gpu * each_gpu: ]    
    else:
        print("Inside division")
        curr_gt_keys = gt_keys[num_gpu * each_gpu: (num_gpu + 1) * each_gpu]
    print(f'Current number of keys: {len(curr_gt_keys)}')
    print("=" * 90)

    curr_gt = {k:v for k,v in gt_questions.items() if k in curr_gt_keys}
    return curr_gt

def iou(outputs, gt, args=None):
    pattern = r'(\d+) to (\d+)'
    matches = re.search(pattern, outputs, re.IGNORECASE)

    if not matches:
        return 0
        
    from_number = float(matches.group(1)) / 100
    to_number = float(matches.group(2)) / 100

    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)

def write_log(log_path, sample_set):
    with open(log_path, 'a') as f:
        f.write(json.dumps(sample_set) + '\n')

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

# Function to find the JSON objects
def find_json_objects(text):
    depth = 0
    start = 0
    objects = []
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                objects.append(text[start:i+1])
    
    return objects

def run_eval(args):
    assert args.mode in ['dvc-capfirst', 'dvc-timefirst', 'grounding', 'single']
    
    if args.mode == 'dvc-capfirst':
        answer_file = os.path.join(args.output_file, 'capfirst.txt')
    elif args.mode == 'dvc-timefirst':
        answer_file = os.path.join(args.output_file, 'timefirst.txt')
    elif args.mode == 'single':
        answer_file = os.path.join(args.output_file, 'single.txt')
    else:
        answer_file = os.path.join(args.output_file, 'grounding.txt')
        
    
    if args.task2:
        if args.mode == 'dvc-capfirst':
            answer_file = os.path.join(args.output_file, 'capfirst_task2.txt')
        elif args.mode == 'dvc-timefirst':
            answer_file = os.path.join(args.output_file, 'timefirst_task2.txt')


    if args.vtimellm: # for vtimellm
        with open(answer_file, 'r') as file:
            file_content = file.read()

        # Find and parse the JSON objects
        json_objects = find_json_objects(file_content)
        logs = []

        for obj in json_objects:
            try:
                logs.append(json.loads(obj))
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON object: {e}")
                
    else:
        logs = []
        with open(answer_file) as f:
            for num, line in enumerate(f):
                try:
                    json_data = json.loads(line)
                    logs.append(json_data)
                
                except Exception as e:
                    print(e, line)

    print(f'Length: {len(logs)}')
    
    if args.mode in ['dvc-capfirst', 'dvc-timefirst', 'single']:
        print("====================== Captioning =====================")
        captioning_metrics(logs, args.question_file, print_matrix=True, args=args)


def run_create(args):
    args.save_file = args.save_file + '_conv'

    cap_first = []
    time_first = []
    cap_first_t2 = []
    time_first_t2 = []
    
    for i in range(args.num_samples):

        if args.task2:
            
            cap_path_t2 =  os.path.join(args.output_file, f'capfirst_score_t2_{i}.json')
            time_path_t2 = os.path.join(args.output_file, f'timefirst_score_t2_{i}.json')   
            
            cap_t2 = json.load(open(cap_path_t2))
            time_t2 = json.load(open(time_path_t2))
            
            cap_first_t2.append(cap_t2)
            time_first_t2.append(time_t2)
        else:
            cap_path = os.path.join(args.output_file, f'capfirst_score_{i}.json')
            time_path = os.path.join(args.output_file, f'timefirst_score_{i}.json')        
            
            cap_ = json.load(open(cap_path))
            time_ = json.load(open(time_path))
            
            cap_first.append(cap_)
            time_first.append(time_)

    # dict_keys(['vid', 'precision', 'recall', 'preference', 'pred_sentences', 'gold_sentences', 'pred_timestamps', 'gold_timestamps', 'original'])
    gt_json = json.load(open(args.question_file))
    create_json(cap_first, time_first, cap_first_t2, time_first_t2, gt_json, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--save-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--create-json', action='store_true')
    
    parser.add_argument('--mode', help='Which inference mode to use', required=True,  choices=['all', 'dvc-capfirst', 'dvc-timefirst', 'grounding', 'single'])
    parser.add_argument('--task2', action='store_true',  help='Whether to also compute for the intermediate dpo preference data',)
    parser.add_argument('--num_samples', type=int, default=2)


    parser.add_argument('--eval_threshold', type=float, default=0, help="threshold for soda difference of two samples (GAP for M-DPO)") 
    parser.add_argument('--save-file-initial', type=str, help='Initial part of the name used for saving (preference data)', default="")
    parser.add_argument('--overwrite_save_directory', type=str, default="")

    # Save file type 
    # parser.add_argument('--wo_gt', action='store_true') # ALWAYS TRUE
    parser.add_argument('--with_CAPFirst', type=int, default=1)
    parser.add_argument('--with_TIMEFirst', type=int, default=1)
    
    # specify datset
    parser.add_argument('--youcook2', action='store_true')

    # specify model
    parser.add_argument('--vtimellm', action='store_true')
    
    args = parser.parse_args()

    
    if args.create_json: # Creating annotation file (for M-DPO)
        run_create(args) 
    else:
        run_eval(args) # Calculating preference based on the metric
