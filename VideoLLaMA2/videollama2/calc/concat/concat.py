import json

# eval_threshold = 10

# # t2_path = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi/conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'
# t3_path_gt = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples/conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'
# t3_path_gt_not = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log//generate-samples-multi-not-5000//conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'
# t3_path_generated = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'

# t3_path_gt = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples/conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'
# t3_path_gt_not = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log//generate-samples-multi-not-5000//conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'
# t3_path_generated = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/conv_dpo_gt_wo_gt_eval{eval_threshold}.0.json'


# t2_path = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/full_time5_cap10conv_dpo_with_gt_3_eval10.0_5.0_gt_5.0_5.0_t2.json'
# t3_path_gt_not = "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/full_time5_cap10conv_dpo_with_gt_3_eval10.0_5.0_gt_5.0_5.0.json"
# t3_path_generated = "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/full_time5_cap10conv_dpo_with_gt_3_eval10.0_5.0_gt_5.0_5.0.json"


# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINALconv_dpo_with_gt_3_eval10.0_5.0_gt_5.0_5.0_t2.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINALconv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/FINALconv_dpo_gt_wo_gt_eval10.0_5.0_gt_5.0_5.0.json


t2_path = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINALconv_dpo_with_gt_3_eval10.0_5.0_gt_5.0_5.0_t2.json'
t3_path_generated = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/FINALconv_dpo_gt_wo_gt_eval10.0_5.0_gt_5.0_5.0.json'
t3_path_gt_not = f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINALconv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json'
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINALconv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json


t2 = json.load(open(t2_path))
# t3_gt = json.load(open(t3_path_gt))
t3_gt_not = json.load(open(t3_path_gt_not))
t3_gt_generated = json.load(open(t3_path_generated))


# print([len(d['vid']) for d in [t2, t3_gt, t3_gt_not, t3_gt_generated]])
print([len(d['vid']) for d in [t2, t3_gt_not, t3_gt_generated]])


# final_dict  = {k:t3_gt_generated[k]+t2[k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
# final_dict  = {k:t3_gt_generated[k]+t2[k]+t3_gt[k]+t3_gt_not[k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
final_dict  = {k:t2[k]  + t3_gt_generated[k] + t3_gt_not[k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
# final_dict  = {k:t3_gt_generated[k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}

# final_dict = {}
# final_dict['prompt'] = t3_1['prompt'] + t3_2['prompt'] + t2['prompt']
# final_dict['chosen'] = t3_1['chosen'] + t3_2['chosen'] + t2['chosen']
# final_dict['rejected'] = t3_1['rejected'] + t3_2['rejected'] + t2['rejected']
# final_dict['vid'] = t3_1['vid'] + t3_2['vid'] + t2['vid']
# final_dict['logits'] = t3_1['logits'] + t3_2['logits'] + t2['logits']


# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_gt_2.json
# VideoLLaMA2/outputs/log/Final-generate-samples/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/capfirst_score_conv_dpo1.json
print([len(v) for i,v in final_dict.items()])


breakpoint()
breakpoint()
# with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-wo-gt-btw-generated-{eval_threshold}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-FINAL.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))

breakpoint()

# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt_eval10.0.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000