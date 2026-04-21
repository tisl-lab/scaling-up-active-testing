import torch
from torch import nn
from metrics import nll, accuracy
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

class Evaluation:
    def __init__(self,
                 model,
                 dataset,
                 loss,
                device):
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.device = device

    def get_dataloader(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def icl(self):
        raise NotImplementedError

    def temperature_scaling(self, set_name, model_file, lr=1e-3, max_iter=2e4, init_val=1.5, verbose=True):
        """Compute optimal temperature from a validation set using gradient descent."""
        scores, targets = utils.load_tensors([f'{self.dataset.dataset_file}/{model_file}/{set_name}_set_scores', 
                                    f'{self.dataset.dataset_file}/{set_name}_set_targets'])
        scores = scores.to(self.device)
        targets = targets.to(self.device)
        # initialization : T=1
        temperature = nn.Parameter(torch.ones(1).to(self.device) * init_val)
        optimiser = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        # optimisation
        def eval_():
            optimiser.zero_grad()
            loss = self.loss(scores, targets, weights=None, temperature=temperature)
            loss.backward()
            return loss
        optimiser.step(eval_)
        optimal_temperature = temperature.item()
        optimal_loss = self.loss(scores, targets, weights=None, temperature=optimal_temperature).cpu()
        self.optimal_temperature, self.optimal_loss = optimal_temperature, optimal_loss
        # saving results
        utils.save_tensors([torch.tensor(optimal_temperature), optimal_loss], 
                           [f'{self.dataset.dataset_file}/{model_file}/{set_name}_set_temperature',
                            f'{self.dataset.dataset_file}/{model_file}/{set_name}_set_optimal_loss'])
        if verbose:
            print(f'Temperature: {temperature} - Corresponding loss: {optimal_loss}')

class EvaluationClassification(Evaluation):
    """Evaluation class for classification tasks."""
    def __init__(self,
                 model,
                 dataset,
                device):
        super(EvaluationClassification, self).__init__(model, dataset, nll, device)
        if ('llama' in self.model.model_file.lower() or "gemma" in self.model.model_file.lower()) and (self.dataset.dataset_file == 'subjective'):
            # subjective tokenized as subject + ive
            self.model.get_tokens([self.dataset.labels[0], 'subject'])
        elif ('phi' in self.model.model_file.lower()) and (self.dataset.dataset_file == 'subjective'):
            # subjective tokenized as subject + ive
            # objective tokenized as object + ive
            self.model.get_tokens(['object', 'subject'])
        else:
            self.model.get_tokens(self.dataset.labels)

    def get_dataloader(self, set_name, batch_size):
        dataset = self.dataset.get(set_name)
        return DataLoader(dataset, 
                          batch_size=batch_size, 
                          collate_fn=self.dataset.data_collator, 
                          shuffle=False
                            )

    def get_scores(self, set_name):
        return utils.load_tensors(f'{self.dataset.dataset_file}/{self.model.filename}/{set_name}_set_scores')

    def get_labels(self, set_name):
        return self.dataset.get_labels(set_name)

    def evaluate(self, set_name, batch_size):
        """Model evaluation."""
        eval_dataloader = self.get_dataloader(set_name, batch_size)
        with torch.no_grad():
            res_scores = torch.zeros((0, self.dataset.num_classes))
            # batch evaluation
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model.model.model.generate(**batch,
                                                            max_new_tokens=self.model.max_new_tokens,
                                                            return_dict_in_generate=True,
                                                            output_scores=True,
                                                            output_logits=True,
                                                            temperature=1.,
                                                            do_sample=True,
                                                            pad_token_id=self.model.model.tokenizer.eos_token_id,
                                                            # Otherwise top_k=50 and top_p=0.9 by default.
                                                            top_k=None,
                                                            top_p=None)
                res_scores = torch.concat(
                    (res_scores, 
                    outputs.scores[0][:, self.model.tokens].cpu()
                    ), dim=0)
        return res_scores
        
    def evaluation_procedure(self, set_name, batch_size, verbose=True, save=True):
        """Evaluate model and compute corresponding loss and accuracy."""
        # model evaluation
        self.res_scores = self.evaluate(set_name, batch_size)
        self.targets = self.get_labels(set_name)
        new_dir = f'{self.dataset.dataset_file}/{self.model.model_file}'
        if save:
            utils.save_tensors(self.res_scores, 
                               f'{new_dir}/{set_name}_set_scores',
                              add_duplicates=True)
        self.accuracy = accuracy(self.res_scores, self.targets)
        self.final_loss = nll(self.res_scores, self.targets)
        if save:
            utils.save_tensors([self.accuracy, self.final_loss],
                       [f'{new_dir}/{set_name}_set_accuracy',
                        f'{new_dir}/{set_name}_set_loss'],
                              add_duplicates=True)
        if verbose:
            print(f'Accuracy: {self.accuracy} - Loss: {self.final_loss}')

    def icl_prompt(self, sentence, dataset, input_prompt, additional_dataset=None):
            """Create prompt with in-context examples."""
            label = self.dataset.labels
            for s, l in zip(dataset['sentence'], dataset['label']):
                input_prompt += f"""Sentence: '{s}' \n{self.dataset.answer_string}: {label[l]}\n\n"""
            if additional_dataset is not None:
                for s, l in zip(additional_dataset['sentence'], additional_dataset['label']):
                    input_prompt += f"""Sentence: '{s}' \n{self.dataset.answer_string}: {label[l]}\n\n"""
            input_prompt += f"""Sentence: '{sentence}' \n{self.dataset.answer_string}:"""
            return input_prompt

    def evaluate_icl(self, set_name, icl_prompt, batch_size=1, set_size=None):
        """Model evaluation with in-context learning."""
        # icl examples are included in the function icl_prompt
        dataset = self.dataset.raw_dataset[self.dataset.get_indices(set_name)]['sentence']
        if set_size is not None:
            dataset = dataset[:set_size]
        with torch.no_grad():
            res_scores = torch.zeros((0, self.dataset.num_classes))
            l = len(dataset)
            for batch in tqdm(range(0, l, batch_size)):
                datapoints = [icl_prompt(sentence) for sentence in dataset[batch:min(l, batch_size + batch)]]
                data = self.model.model.tokenizer(datapoints, 
                                                  return_tensors='pt', 
                                                  padding=True, 
                                                  truncation=False).to(self.device)
                outputs = self.model.model.model.generate(**data,
                                                            max_new_tokens=self.model.max_new_tokens,
                                                            return_dict_in_generate=True,
                                                            output_scores=True,
                                                            output_logits=True,
                                                            temperature=1.,
                                                            do_sample=True,
                                                            pad_token_id=self.model.model.tokenizer.eos_token_id,
                                                            # Otherwise top_k=50 and top_p=0.9 by default.
                                                            top_k=None,
                                                            top_p=None,)
                res_scores = torch.concat(
                    (res_scores, 
                     outputs.scores[0][:, self.model.tokens].cpu()
                    ), dim=0)
        return res_scores

    def evaluation_icl_procedure(self, set_name, nb_icl_examples, batch_size=1, version=0, verbose=True, save=True):
        """Evaluate model with in-context learning and compute loss and accuracy."""
        icl_set = utils.load_tensors(f'{self.dataset.dataset_file}/icl_set_indices')
        v_format = f'_v{version}' if version > 0 else ''
        selected_examples = utils.load_tensors(
            f'{self.dataset.dataset_file}/icl{nb_icl_examples}_selected_examples{v_format}')
        new_dir = f'{self.dataset.dataset_file}/{self.model.model_file}_icl{nb_icl_examples}{v_format}'
        # prompt with in-context examples
        icl_prompt = lambda sentence: self.icl_prompt(sentence,
                                                      self.dataset.icl_set[icl_set[selected_examples]],
                                                      self.model.prompt)
        # model evaluation
        self.res_scores = self.evaluate_icl(set_name, icl_prompt, batch_size)
        self.targets = self.get_labels(set_name)
        if save:
            if not(os.path.isdir(new_dir)):
                os.mkdir(new_dir)
            utils.save_tensors(self.res_scores, 
                               f'{new_dir}/{set_name}_set_scores',
                               add_duplicates=True)
        self.accuracy = accuracy(self.res_scores, self.targets)
        self.final_loss = nll(self.res_scores, self.targets)
        if save:
            utils.save_tensors([self.accuracy, self.final_loss],
                               [f'{new_dir}/{set_name}_set_accuracy',
                                f'{new_dir}/{set_name}_set_loss'], 
                                add_duplicates=True)
        if verbose:
            print(f'Accuracy: {self.accuracy} - Loss: {self.final_loss}')

    def get_results(self, set_name, version=0, icl=(False, 0)):
        """Get results from saved files."""
        v_format = f'_v{version}' if version > 0 else ''
        if icl[0]:
            accuracy, loss_value = utils.load_tensors([
        f'{self.dataset.dataset_file}/{self.model.model_file}_icl{icl[1]}{v_format}/{set_name}_set_accuracy',
        f'{self.dataset.dataset_file}/{self.model.model_file}_icl{icl[1]}{v_format}/{set_name}_set_loss'
        ])
            return accuracy, loss_value
        else:
            accuracy, loss_value = utils.load_tensors([
                f'{self.dataset.dataset_file}/{self.model.model_file}/{set_name}_set_accuracy',
                f'{self.dataset.dataset_file}/{self.model.model_file}/{set_name}_set_loss'
            ])
            return accuracy, loss_value
        
        

    