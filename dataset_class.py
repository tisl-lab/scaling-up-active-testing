from datasets import load_dataset
from transformers import DataCollatorWithPadding
import utils
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

class BaseDataset:
    def __init__(self,
                dataset_file, 
                 num_classes=2):
        self.dataset_file = dataset_file
        self.num_classes = num_classes
        if not(os.path.isdir(self.dataset_file)):
            os.mkdir(self.dataset_file)
        self.subsets = {}
        self.get_existing_subsets()

    def select_icl_examples(self, nb_examples):
        indices = utils.load_tensors(f'{self.dataset_file}/icl_set_indices')
        selected = []
        for file in os.listdir(self.dataset_file):
            if file.startswith('icl') and 'selected_examples' in file:
                filename = file.split('.pt')[0]
                selected.append(utils.load_tensors(f'{self.dataset_file}/{filename}'))
        if len(selected):
            selected = torch.concat(selected)
        else:
            selected = torch.tensor([])
        selection = torch.randperm(len(indices))
        selection = selection[~torch.isin(selection, selected)][:nb_examples]
        if len(selection) < nb_examples:
            print('Warning: not enough examples.')
        utils.save_tensors(selection, 
                           f'{self.dataset_file}/icl{nb_examples}_selected_examples', 
                           add_duplicates=True)

    def get_subset(self, set_name, dataset):
        assert set_name in self.subsets.keys(), "Subset not created."
        indices = utils.load_tensors(f'{self.dataset_file}/{set_name}_set_indices')
        return Subset(dataset, indices)

    def get_indices(self, set_name):
        return utils.load_tensors(f'{self.dataset_file}/{set_name}_set_indices')

    def get_labels(self, set_name):
        return utils.load_tensors(f'{self.dataset_file}/{set_name}_set_targets')

    def get_existing_subsets(self):
        for file in os.listdir(self.dataset_file):
            if file.endswith('set_indices.pt'):
                self.subsets[file.split('_set_indices.pt')[0]] = len(torch.load(f'{self.dataset_file}/{file}', weights_only=False))

    def load(self):
        raise NotImplementedError
        
    def create_subset(self, new_set_name, set_length, dataset, label='label'):
        assert not(new_set_name in self.subsets.keys()), "Subset already exists."
        used_indices = []
        for set_name, length in self.subsets.items():
            indices = utils.load_tensors(f'{self.dataset_file}/{set_name}_set_indices').tolist()
            used_indices = used_indices + indices
        available_indices = torch.tensor(np.setdiff1d(np.arange(len(dataset)), used_indices))
        available_set = Subset(dataset, available_indices)
        length = len(available_set)
        new_subset, _ = torch.utils.data.random_split(available_set, [set_length/length, 1-set_length/length])
        corresponding_indices = available_indices[new_subset.indices]
        self.subsets[new_set_name] = len(corresponding_indices)
        utils.save_tensors(corresponding_indices, f'{self.dataset_file}/{new_set_name}_set_indices')
        targets = torch.tensor(dataset[corresponding_indices][label])
        if len(targets.shape) == 1:
            t = torch.zeros(len(targets), self.num_classes)
            t[torch.arange(len(t)), targets] = 1
            targets = t
        utils.save_tensors(targets, f'{self.dataset_file}/{new_set_name}_set_targets')

class SST2Dataset(BaseDataset):
    def __init__(self,
                 dataset_file):
        super(SST2Dataset, self).__init__(dataset_file)
        self.raw_dataset = load_dataset("glue", "sst2", trust_remote_code=True)['train']
        self.labels = ['negative', 'positive']
        self.answer_string = 'Label'
        self.icl_set = self.raw_dataset

    def load(self, prompt, tokenizer, device):
        def prompt_function(sentence):
            return f"""{prompt}Sentence: '{sentence}' \n{self.answer_string}:"""
        def tokenize_function(example):
            return tokenizer(example['sentence'], return_tensors='pt', padding=True, truncation=False).to(device)
        raw_datasets_prompt = self.raw_dataset.map(lambda d: {'sentence': prompt_function(d['sentence']), 
                                                              'label': d['label'], 
                                                              'idx': d['idx']})
        tokenized_datasets = raw_datasets_prompt.map(lambda example: tokenize_function(example), batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'idx'])
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def get(self, subset_name):
        return self.get_subset(subset_name, self.tokenized_datasets)

class SubjectiveDataset(BaseDataset):
    def __init__(self,
                 dataset_file):
        super(SubjectiveDataset, self).__init__(dataset_file, num_classes=2)
        self.raw_dataset = load_dataset('SetFit/subj', trust_remote_code=True)['train'].rename_column('text', 'sentence')
        self.labels = ['objective', 'subjective']
        self.answer_string = 'Answer'
        self.icl_set = load_dataset("SetFit/subj", trust_remote_code=True)['test'].rename_column('text', 'sentence')

    def load(self, prompt, tokenizer, device):
        def prompt_function(sentence):
            return f"""{prompt}Sentence: '{sentence}' \n{self.answer_string}:"""
        def tokenize_function(example):
            return tokenizer(example['sentence'], 
                             return_tensors='pt', 
                             padding=True, 
                             truncation=False).to(device)
        raw_datasets_prompt = self.raw_dataset.map(lambda d: {'sentence': prompt_function(d['sentence']), 
                                                              'label': d['label']})
        tokenized_datasets = raw_datasets_prompt.map(lambda example: tokenize_function(example), batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'label_text'])
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def get(self, subset_name):
        return self.get_subset(subset_name, self.tokenized_datasets)

class HateSpeechDataset(BaseDataset):
    def __init__(self,
                 dataset_file):
        super(HateSpeechDataset, self).__init__(dataset_file, num_classes=2)
        self.raw_dataset = load_dataset('hate_speech18', trust_remote_code=True)['train'].rename_column('text', 'sentence')
        # filter labels that are not 0 or 1
        self.raw_dataset = self.raw_dataset.filter(lambda example: example['label'] < 2)
        self.labels = ['no', 'yes']
        self.answer_string = 'Answer'
        self.icl_set = self.raw_dataset

    def load(self, prompt, tokenizer, device):
        def prompt_function(sentence):
            return f"""{prompt}Sentence: '{sentence}' \n{self.answer_string}:"""
        def tokenize_function(example):
            return tokenizer(example['sentence'], 
                             return_tensors='pt', 
                             padding=True, 
                             truncation=False).to(device)
        raw_datasets_prompt = self.raw_dataset.map(lambda d: {'sentence': prompt_function(d['sentence']), 
                                                              'label': d['label']})
        tokenized_datasets = raw_datasets_prompt.map(lambda example: tokenize_function(example), batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'subforum_id', 'num_contexts', 'user_id'])
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def get(self, subset_name):
        return self.get_subset(subset_name, self.tokenized_datasets)


class AGNewsDataset(BaseDataset):
    def __init__(self,
                 dataset_file):
        super(AGNewsDataset, self).__init__(dataset_file, num_classes=4)
        self.raw_dataset = load_dataset('ag_news', trust_remote_code=True)['train'].rename_column('text', 'sentence')
        self.labels = ['world', 'sports', 'business', 'science and technology']
        self.answer_string = 'Answer'
        self.icl_set = self.raw_dataset

    def load(self, prompt, tokenizer, device):
        def prompt_function(sentence):
            return f"""{prompt}Sentence: '{sentence}' \n{self.answer_string}:"""
        def tokenize_function(example):
            return tokenizer(example['sentence'], 
                             return_tensors='pt', 
                             padding=True, 
                             truncation=False).to(device)
        raw_datasets_prompt = self.raw_dataset.map(lambda d: {'sentence': prompt_function(d['sentence']), 
                                                              'label': d['label']})
        tokenized_datasets = raw_datasets_prompt.map(lambda example: tokenize_function(example), batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def get(self, subset_name):
        return self.get_subset(subset_name, self.tokenized_datasets)

class FPBDataset(BaseDataset):
    def __init__(self,
                 dataset_file):
        super(FPBDataset, self).__init__(dataset_file, num_classes=3)
        self.raw_dataset = load_dataset('financial_phrasebank', 'sentences_allagree', trust_remote_code=True)['train']
        self.labels = ['negative', 'neutral', 'positive']
        self.answer_string = 'Answer'
        self.icl_set = self.raw_dataset

    def load(self, prompt, tokenizer, device):
        def prompt_function(sentence):
            return f"""{prompt}Sentence: '{sentence}' \n{self.answer_string}:"""
        def tokenize_function(example):
            return tokenizer(example['sentence'], 
                             return_tensors='pt', 
                             padding=True, 
                             truncation=False).to(device)
        raw_datasets_prompt = self.raw_dataset.map(lambda d: {'sentence': prompt_function(d['sentence']), 
                                                              'label': d['label']})
        tokenized_datasets = raw_datasets_prompt.map(lambda example: tokenize_function(example), batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def get(self, subset_name):
        return self.get_subset(subset_name, self.tokenized_datasets)


class MMLUDataset(BaseDataset):
    def __init__(self,
                 dataset_file):
        super(MMLUDataset, self).__init__(dataset_file, num_classes=4)
        self.raw_dataset = load_dataset('cais/mmlu', 'all', trust_remote_code=True)['test']
        self.labels = ['A', 'B', 'C', 'D']
        self.answer_string = 'Answer'
        self.icl_set = load_dataset('cais/mmlu', 'all', trust_remote_code=True)['auxiliary_train']

    def load(self, prompt, tokenizer, device):
        def prompt_function(sentence, choices):
            list_choices = ""
            for i in range(len(choices)):
                list_choices += f"{self.labels[i]}. {choices[i]}\n"
            return f"""{prompt}Question: {sentence} \n{list_choices}{self.answer_string}:"""
        def tokenize_function(example):
            return tokenizer(example['question'],
                             return_tensors='pt',
                             padding=True,
                             truncation=False).to(device)
        raw_datasets_prompt = self.raw_dataset.map(lambda d: {'sentence': prompt_function(d['question'], d['choices']), 
                                                              'label': d['label']})
        tokenized_datasets = raw_datasets_prompt.map(lambda example: tokenize_function(example), batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def get(self, subset_name):
        return self.get_subset(subset_name, self.tokenized_datasets)

        