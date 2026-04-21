import copy
import logging
from collections import Counter
import torch
import os

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from transformers import set_seed

from huggingface_hub import snapshot_download

class N:
    LOGITS = 'logits'
    LOG_LIKELIHOODS = 'log_likelihoods'
    ADD = '_unwarped'
    LOGITS_UNWARPED = f'{LOGITS}{ADD}'
    LOG_LIKELIHOODS_UNWARPED = f'{LOG_LIKELIHOODS}{ADD}'


class HuggingfaceModel:
    """HuggingfaceModel."""

    def __init__(
            self, *, name, random_seed, debug, stop_sequences, max_new_tokens,
            temperature, top_k, top_p,
    ):
        set_seed(random_seed)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.name = name
        self.max_new_tokens = max_new_tokens

        if 'llama' in name.lower():

            if name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                name = name[:-len('-8bit')]
                autolm = True
            elif name.endswith('-16bit'):
                kwargs = {'torch_dtype': torch.bfloat16}
                name = name[:-len('-16bit')]
                autolm = True
            else:
                kwargs = {}
                autolm = False

            if 'Llama-2' in name:
                base = 'meta-llama'
                name = name + '-hf'
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{name}", device_map="auto",
                token_type_ids=None)

            llama65b = '65b' in name and base == 'huggyllama'
            llama2_70b = '70b' in name and base == 'meta-llama'


            if ('7b' in name or '13b' in name) or autolm:
                if autolm:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        f"{base}/{name}", **kwargs,
                        low_cpu_mem_usage=True)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        f"{base}/{name}", device_map="auto",
                        max_memory={0: '80GIB'}, **kwargs,
                        low_cpu_mem_usage=True,
                        attn_implementation="sdpa")

            elif llama2_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{name}")
                #config.load_in_8bit = True
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation="sdpa")
                self.model.tie_weights()
                max_mem = 15 * 4686198491

                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16',
                    no_split_module_classes=['LlamaDecoderLayer']
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {
                    f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                # get snapshot folder
                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map="auto",
                    dtype='float16',
                    skip_keys='past_key_values',
                    )

                for i in self.model.named_parameters():
                    logging.info('%s -> %s', i[0], i[1].device)

            else:
                raise ValueError

        elif 'falcon' in name:
            model_id = f'tiiuae/{name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        elif 'phi' in name.lower():
            model_id = f'microsoft/{name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True,
                device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",)

        elif 'gpt' in name.lower():
            model_id = f'openai-community/{name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True,
                device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",)

        elif 'gemma' in name.lower():
            
            model_id = f'google/{name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True,
                device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",)
            
        else:
            raise ValueError

        self.model.eval()
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in name else (1024 if 'gpt' in name.lower() else 2048)
        self.forward_pass_hooks = set()
        if 'Llama-2-7b' in name:
            self.last_hidden_dim = 4096
        elif 'Phi-2' in name:
            self.last_hidden_dim = 2560
        elif 'Phi-3' in name:
            self.last_hidden_dim = 3072
        elif 'Llama-2-70b' in name:
            self.last_hidden_dim = 8192
        else:
            self.last_hidden_dim = None
            
    def predict(self, input_data, *, temperature=None, return_full=False, hooks=None):
        out = dict()

        if temperature is None:
            temperature = self.temperature

        if hooks is None:
            hooks = self.forward_pass_hooks

        out['temperature'] = temperature

        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if 'llama' in self.name.lower() or 'falcon' in self.name or 'phi' in self.name:
            if 'token_type_ids' in inputs:  # seems to have been updated
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                # Otherwise top_k=50 and top_p=0.9 by default.
                top_k=self.top_k,
                top_p=self.top_p,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        # Only skip the bos token. We want to keep the eos token, s.t. we can
        # detect it in `stop_at` calculation below.
        # NOTE: This is an improvement we may want to add to semantic_entropy
        # codebase as well.
        output_sequences = outputs.sequences[0]
        if output_sequences[0] == self.tokenizer.bos_token_id:
            output_sequences = output_sequences[1:]
        full_answer = self.tokenizer.decode(
            output_sequences, skip_special_tokens=False)

        if return_full:
            out['text'] = full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            assert all([stop not in sliced_answer for stop in self.stop_sequences])

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        if not return_full:
            out['text'] = sliced_answer
        # logging.info('Generation for temperature `%.2f` is `%s`.', temperature, sliced_answer)

        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: We do not want this to be the stop token.

        # outputs.hidden_state is a tuple of len = n_generated_tokens.
        # The first hidden state is for the input tokens and is of shape (n_layers) x (batch_size, input_size, hidden_size).
        # (Note this includes the first generated token!)
        # The remaining hidden states are for the remaining generated tokens and is of shape (n_layers) x (batch_size, 1, hidden_size).

        # TODO: Is this note wrong? Or is it true for falcon models?
        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size). We do not get
        # embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.
        if 'full_hidden_states' in hooks:

            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states

            out['full_hidden_states'] = outputs.hidden_states

        if 'last_hidden_state' in hooks:

            # # P_IK wants last state of input. OR DOES IT? DISABLE FOR NOW?
            # # First access states for input
            # last_input = hidden[0]
            # First access states for last token generation before stop token.and
            last_input = hidden[n_generated - 1]
            # Then access last layer for input
            last_layer = last_input[-1]
            # Then access last token in input.
            last_hidden_state = last_layer[:, -1, :].cpu()
            out['last_hidden_state'] = last_hidden_state

        if N.LOGITS in hooks:
            out[N.LOGITS] = [s.detach().cpu().numpy()[0] for s in outputs.scores[:n_generated]]

        if N.LOG_LIKELIHOODS in hooks:
            # Get log_likelihoods.
            # outputs.scores are the logits for the generated token.
            # outputs.scores is a tuple of len = n_generated_tokens.
            # Each entry is shape (bs, vocabulary size).
            # outputs.sequences is the sequence of all tokens: input and generated.
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True)
            # transition_scores[0] only contains the scores for the first generated tokens.
            log_likelihoods = [score.item() for score in transition_scores[0]][:n_generated]

            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')
            #     max_token = True
            # else:
            #     max_token = False
            if len(log_likelihoods) == 0:
                raise ValueError

            out['log_likelihoods'] = log_likelihoods

        if (N.LOGITS_UNWARPED in hooks) or (N.LOG_LIKELIHOODS_UNWARPED in hooks):
            # Get raw logits from extra forward pass. Otherwise, warping (temperature/top_p/)
            re_input = {
                'input_ids': outputs.sequences,
                'attention_mask': torch.ones_like(outputs.sequences)}
            with torch.no_grad():
                re_output = self.model(**re_input)
                # Bring into expected shape.
                re_scores = [s for s in re_output['logits'].transpose(0, 1)]
                # Cut off input. (Adjust for next token prediction offset.)
                re_scores = re_scores[-len(outputs['scores'])-1:]
                # Cut off n_generated. (Adjust for next token prediction offset.)

            if N.LOGITS_UNWARPED in hooks:
                # sub = 1 if max_token else 0
                # out[LOGITS_UNWARPED] = [s.detach().cpu().numpy()[0] for s in re_scores[:n_generated - sub]]
                out[N.LOGITS_UNWARPED] = [s.detach().cpu().numpy()[0] for s in re_scores[:n_generated]]
                assert len(out[N.LOGITS_UNWARPED]) == len(out[N.LOGITS])

            if N.LOG_LIKELIHOODS_UNWARPED in hooks:
                re_log_lik = self.model.compute_transition_scores(
                    outputs.sequences, re_scores[:-1], normalize_logits=True)
                out[N.LOG_LIKELIHOODS_UNWARPED] = [lik.item() for lik in re_log_lik[0][:n_generated]]
                assert len(out[N.LOG_LIKELIHOODS_UNWARPED]) == len(out[N.LOG_LIKELIHOODS])

        # For debugging purposes:
        # Can compare self.tokenizer.encode(sliced_answer) to len(log_likelihoods).
        # But can be off by one due to whitespaces.

        # falcon-7b
        # len(hidden), len(hidden[0]), hidden[0][0].shape, hidden[1][0].shape, hidden[1][-1].shape, hidden[token_stop_index - 1][-1].shape
        # (4, 33, torch.Size([1, 53, 4544]), torch.Size([1, 54, 4544]), torch.Size([1, 54, 4544]), torch.Size([1, 55, 4544]))
        # llama-7b
        # (3, 33, torch.Size([1, 60, 4544]), torch.Size([1, 61, 4544]), torch.Size([1, 61, 4544]), torch.Size([1, 61, 4544]))

        return out


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    logging.info('Device map before split %s', device_map_in)

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                logging.info(f'pop %s', name)
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    logging.info('Device map after split %s', device_map)

    return device_map
	