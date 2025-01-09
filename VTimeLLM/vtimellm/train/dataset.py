import random
import os
import copy
import sys
import json
import torch
import transformers
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


sys.path.append('./')
from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_SEGMENT_TOKEN
from vtimellm import conversation as conversation_lib
from vtimellm.mm_utils import tokenizer_image_token
import pickle
from tqdm import tqdm
from glob import glob
import re

from trl.trl.trainer.utils import DPODataCollatorWithPadding

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    feat_folder: Optional[str] = field(default=None)
    num_bins: int=field( default = 100,metadata={"help": "Number of bins for videoo"})
    data_folder: Optional[str] = field(default=None)

    image_aspect_ratio: str = 'pad'


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_glm(
    sources,
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    
    input_ids = []
    targets = []

    for source in sources:
        tokens, loss_masks = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")], [0, 0]
        def _update(_tokens: List[int], value: int = 1):
            value = int(value)
            tokens.extend(_tokens)
            loss_masks.extend([value] * len(_tokens))
        
        for conv in source:
            if conv["from"] == 'human':
                role_token = tokenizer.get_command("<|user|>")
                loss = False
            else:
                role_token = tokenizer.get_command("<|assistant|>")
                loss = True
                
            token_id = [role_token] + tokenizer_image_token(conv['value'], tokenizer)[2:]
            _update(token_id, loss)
        _update([tokenizer.eos_token_id], False)

        loss_masks = [False] + loss_masks[:-1]
        labels = [(t if m else IGNORE_INDEX) for t, m in zip(tokens, loss_masks)]

        input_ids.append(tokens)
        targets.append(labels)

        # print("Sanity Check >>>>>>>>>>>>>")
        # for t, m in zip(tokens, labels):
        #     decoded =  tokenizer.tokenizer.index_special_tokens[t] \
        #         if t in tokenizer.tokenizer.index_special_tokens \
        #         else tokenizer.decode([t])
        #     print("%20s: %6d -> %6d" % (repr(decoded), t, m))
        # print("<<<<<<<<<<<<< Sanity Check")

    return dict(
        input_ids=torch.tensor(input_ids),
        labels=torch.tensor(targets),
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        all_inputs = []
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                curr_tokenizer = tokenizer_image_token(rou, tokenizer)
                if i == len(rounds) - 2 and curr_tokenizer[-1] != 2:
                    curr_tokenizer  = curr_tokenizer + [2]

                round_len = len(curr_tokenizer)
                all_inputs.append(curr_tokenizer)
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2

            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:

                all_inputs = torch.cat([torch.tensor(i) for i in all_inputs])
                if len(all_inputs) != len(input_ids[0]):
                    # target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. serious issue."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n'
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args

        self.stage4 = True if self.list_data_dict[0]['source'] == 'activitynet' or self.list_data_dict[0]['source'] == 'youcook2' else False

        if not self.stage4:
            features_path = glob(f'{self.data_args.feat_folder}/*.npy')
            features_path = [f.split('/')[-1].replace('.npy', '') for f in features_path]
        else:
            self.features = torch.load(self.data_args.feat_folder)
            features_path = list(self.features.keys())
        
            new_data_dict = []
            for i in tqdm((self.list_data_dict)):
                curr_id = i['id']
                if curr_id in features_path:
                    new_data_dict.append(i)
            
            self.list_data_dict = new_data_dict
        print(f"Number of samples is {len(self.list_data_dict)}")


    def __len__(self):
        # return 50
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(self.list_data_dict[i])

        data_type = 'video'

        if '<image>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<image>', '<video>')
            data_type = 'image'


        if 'meta' in source:
            def convert(duration, x):
                x = x / duration * 100
                x = str(min(round(x), 99))


                if len(x) == 1:
                    x = "0" + x
                return x

            replace_set = []
            for k, v in source['meta']['token'].items():
                replace_set.append((k, convert(source['meta']['duration'], v)))

            for l in range(len(source['conversations'])):
                for x1, x2 in replace_set:
                    source['conversations'][l]['value'] = source['conversations'][l]['value'].replace(x1, x2)

        image = torch.zeros((100 if data_type == 'video' else 1, 768), dtype=torch.float16)

        if not self.stage4:
            try:
                feature_path = '{}/{}.npy'.format(self.data_args.feat_folder, source['id'])
                image = np.load(feature_path) # <N, 768> float16
                image = torch.from_numpy(image)
                if data_type == 'image' and len(image.shape) == 1: # <768>
                    image = image.unsqueeze(0)
            except Exception as e:
                print(e)
                return random.choice(self)

        else:
            try:
                image = self.features[source['id']]
                if data_type == 'image' and len(image.shape) == 1: # <768>
                    image = image.unsqueeze(0)

            except Exception as e:
                print(e)
                return random.choice(self)


        if getattr(self.tokenizer, 'name', None) == 'GLMTokenizer':
            data_dict = preprocess_glm([source["conversations"]], self.tokenizer)
        else:
            data_dict = preprocess(
                [source["conversations"]],
                self.tokenizer,
                has_image=True)
            
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['image'] = image

        if len(data_dict['input_ids']) > self.tokenizer.model_max_length:
            print(f'{data_dict["id"]}, over max sequence')
            
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


class LazySupervisedDPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDPODataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.finetune = False
        self.preloaded = False

        data_len = len(self.list_data_dict['vid'])
        self.features = torch.load(self.data_args.feat_folder)
        self.features_path = list(self.features.keys())

        if 'activitynet' in data_args.feat_folder:
            print("Inside activitynet")
            self.finetune = True
            self.preloaded = True
            glob_features = {i:i for i in self.features.keys()}

        if 'youcook2' in data_args.feat_folder.lower():
            print("Inside YouCook2")
            self.finetune = True
            self.preloaded = True
            glob_features = {i:i for i in self.features.keys()}

        new_data_dict = []
        for i in tqdm(range(data_len)):
            cur_vid = self.list_data_dict['vid'][i]
            if cur_vid in glob_features.keys():
                curr_data = {
                    'prompt': self.list_data_dict['prompt'][i],
                    'chosen': self.list_data_dict['chosen'][i],
                    'rejected': self.list_data_dict['rejected'][i],
                    'video': glob_features[cur_vid],
                    'video_id': cur_vid,
                }
                new_data_dict.append(curr_data)

        self.list_data_dict = new_data_dict
        

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = copy.deepcopy(self.list_data_dict[i])
        MODAL_list = []


        video_file = self.list_data_dict[i]['video']
        video_id = self.list_data_dict[i]['video_id']
        MODAL_list.append('VIDEO')

        if self.preloaded:
            try:
                video = self.features[video_id]
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
        else:
            assert False, 'Video feature should be preloaded'

        data_dict['video'] = video
        data_dict['MODAL_list'] = MODAL_list

        return data_dict


@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(DPODataCollator, self).__init__()
        self.tokenizer = tokenizer

    def collate(self, batch):

        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):

                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ['chosen_input_ids', 'rejected_input_ids']:
            attn_k = k.replace('input_ids', 'attention_mask')
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)
        return padded_batch

    def tokenize_batch_element(
            self,
            prompt: str,
            chosen: str,
            rejected: str,
            MODAL_list: str = None
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        # print([prompt] + chosen)
        # print([prompt] + rejected)
        chosen_sources = [prompt] + chosen  # already in converstaion format
        rejected_sources = [prompt] + rejected
        chosen_data_dict = preprocess([chosen_sources], self.tokenizer)
        # chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = preprocess([rejected_sources], self.tokenizer)
        # rejected_data_dict['attention_mask'] = rejected_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_data_dict = {k: v[0] for k, v in rejected_data_dict.items()}

        for k, toks in {
            "chosen": chosen_data_dict,
            "rejected": rejected_data_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        return batch

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        tokenized_batch = []
        Xs, keys = [], []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            MODAL_list = feature['MODAL_list']
            Xs.append(feature[MODAL_list[0].lower()])
            keys.append(MODAL_list[0].lower())

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, MODAL_list=MODAL_list)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch = self.collate(tokenized_batch)
        padded_batch['images'] = [Xs, keys]  # we do not change the key's name.
        return padded_batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_supervised_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDPODataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,)

    data_collator = DPODataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)