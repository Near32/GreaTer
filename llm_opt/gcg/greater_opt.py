import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import sys
import random


sys.path.append('..')
from tqdm.auto import tqdm
import torch.nn.functional as F

# from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
# from llm_attacks import get_embedding_matrix, get_embeddings


import sys

sys.path.append('..')
from llm_opt.base.attack_manager import Prompter, MultiPrompter, PromptManager, get_embeddings, \
    get_embedding_matrix


def token_gradients(model, input_ids, goal_slice, input_slice, target_slice, loss_slice, current_pos, valid_tokens,
                    focused_target_slice=None, control_weight=0.2, temperature=0.2):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    goal_slice : slice
        The slice of the input sequence to be used as the goal.
    input_slice : slice (control_slice)
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.
    current_pos : int
        The current position in the input_ids for which gradients are computed.
    valid_tokens : list
        The subset of token indices that are considered valid for embedding.

    Returns
    -------
    torch.Tensor
        The gradients of the token at current_pos with respect to the loss,
        considering only the valid tokens.
    """

    def control_loss(logits, ids, goal_slice, control_slice):

        l_slice = slice(control_slice.start - 1, control_slice.stop - 1)
        temp_slice = slice(control_slice.start, control_slice.stop)

        logsftmaxed = F.log_softmax(logits[:, l_slice, :], dim=2)
        targeted_logsftmax = logsftmaxed[0, torch.arange(logits[:, l_slice, :].shape[1]), ids[temp_slice]]

        return -targeted_logsftmax

    # Ensure current token is in valid tokens
    current_pos = input_slice.start + current_pos
    current_token = input_ids[current_pos].item()


    # Create the reduced embedding matrix
    embed_weights = get_embedding_matrix(model)[valid_tokens]

    # Create one_hot matrix
    one_hot = torch.zeros(
        1, len(valid_tokens),
        device=model.device,
        dtype=embed_weights.dtype
    )
    token_index = (valid_tokens == current_token).nonzero(as_tuple=True)[0].item()
    one_hot[0, token_index] = 1.0
    one_hot.requires_grad_()

    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # Replace embedding at current_pos with the modified one
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat([
        embeds[:, :current_pos, :],
        input_embeds,
        embeds[:, current_pos + 1:, :]
    ], dim=1)

    logits = model(inputs_embeds=full_embeds, return_dict=False)[0]
    goals = input_ids[goal_slice]
    targets = input_ids[target_slice]

    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    if focused_target_slice:
        focused_loss_slice = slice(focused_target_slice.start - 1, focused_target_slice.stop - 1)
        focused_loss = nn.CrossEntropyLoss()((logits[0, focused_loss_slice, :]) / temperature,
                                             input_ids[focused_target_slice])
        loss = control_weight * control_loss(logits, input_ids, goal_slice, input_slice).mean() + focused_loss
    else:
        loss = loss + control_weight * control_loss(logits, input_ids, goal_slice, input_slice).mean()

    loss.backward()
    del embed_weights, input_embeds, embeds, full_embeds, logits; gc.collect()
    torch.cuda.empty_cache()

    return one_hot.grad.clone()


class GCGPrompter(Prompter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def grad(self, model, current_pos, valid_tokens):
        return token_gradients(
            model,
            self.input_ids.to(model.device),
            self._goal_slice,
            self._control_slice,
            self._target_slice,
            self._loss_slice,
            current_pos,
            valid_tokens,
            self._focused_target_slice,

        )


class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pos = 0

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):
        """
        Sample new control tokens based on gradients.
        
        Args:
            grad: Gradient tensor
            batch_size: Number of candidates to generate
            topk: Top-k sampling parameter
            temp: Temperature for sampling
            allow_non_ascii: Whether to allow non-ASCII tokens
            
        Returns:
            Tensor of shape (batch_size, seq_len)
        """

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        top_values = (-grad).topk(topk, dim=1).values
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        # new_token_pos[:] = self.current_pos
        # new_token_val = torch.gather(
        #     top_indices[new_token_pos], 1,
        #     torch.arange(0, batch_size/len(control_toks), device=grad.device).type(torch.int64).repeat(len(control_toks)).unsqueeze(1)
        # )

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,
            torch.randint(0, topk, (batch_size, 1),
                          device=grad.device)
        )

        # new_token_val = torch.gather(
        #     top_indices[new_token_pos], 1,
        #     (torch.arange(0, batch_size , device=grad.device).view(batch_size,1)),
        # )

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        self.current_pos = (self.current_pos + 1) % len(control_toks)
        return new_control_toks


def find_top_p(logits_softmaxed_currpos, top_p, min_keep=10):
    # Sort the logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits_softmaxed_currpos, descending=True)

    # Compute cumulative probabilities of the sorted logits
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create a mask for the cumulative probabilities that are less than or equal to top_p
    sorted_indices_to_keep = cum_probs <= top_p

    # Ensure at least `min_keep` tokens are kept
    if sorted_indices_to_keep.sum().item() < min_keep:
        sorted_indices_to_keep[:min_keep] = True

    return sorted_indices[sorted_indices_to_keep], sorted_logits[sorted_indices_to_keep]


class GCGMultiPrompter(MultiPrompter):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.loss_cache = {}
        self.selection_history = {}
        self.sequential_patience = 0
        self.initial_length = len(self.prompts[0].control_toks)

    def get_grads(self, main_device, current_pos, valid_tokens):

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model, current_pos, valid_tokens)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            grad += new_grad

        return grad

    def morph_control(
        self, 
        workers, 
        top_k=10, 
        top_p=None, 
        accumulate=False, 
        allow_non_ascii=False,
        sequentially_increasing=True, 
        intersection_across_examples=5, 
        sequential_patience_limit=2, 
        num_intersections=3, 
        control_weight=0.2,
    ):
        # only true for a single model
        # take the lm probs for the single aforementioned model
        # accumulate is a flag to determine whether to accumulate the lm probs across the batch of samples or not

        # Stage 1: Some initial Preprocessing

        minimum_keep_for_selection = 3
        main_device = self.models[0].device
        all_intersections = []

        assert (top_k is None) ^ (
                top_p is None), "Constrained Decoding: Both top_k and top_p are either None or not specified."
        ellipsis_token = self.workers[0].tokenizer.encode("_")[-1]

        lm_probs = 0
        topk = top_k
        control_len = len(self.prompts[0].control_toks)
        control_toks = self.prompts[0].control_toks

        if self.prompts[0].current_pos >= control_len:
            self.prompts[0].current_pos = self.prompts[0].current_pos - 1


        if self.prompts[0].current_pos == 0:
            self.sequential_patience += 1

        if self.sequential_patience > sequential_patience_limit:
            sequentially_increasing = True

        ending_token = self.workers[0].tokenizer.encode("sample.")[-1]

        # faster implementation of same thing
        # Makeshift - no need for this loop as it goes only once

        for pos in range(self.prompts[0].current_pos, self.prompts[0].current_pos + 1, 1):
            # for pos in range(control_len):
            # Stage 2: Get the logits for the current position, get candidate tokens by intersection
            lm_topks = []
            logits_softmaxed_cached = []
            cum_logits = 0

            for k, worker in enumerate(workers):
                worker(self.prompts[k], "logits", worker.model, control_toks, return_ids=True)
            logits, ids = zip(*[worker.results.get() for worker in workers])
            our_logit = logits[0]
            for i in range(len(our_logit)):
                logits_softmaxed = F.log_softmax(our_logit[i][:, (self.prompts[0][i]._control_slice.start - 1):(
                        self.prompts[0][i]._control_slice.stop - 1), :], dim=2)
                logits_softmaxed_currpos = logits_softmaxed[0, pos, :]

                del logits_softmaxed; gc.collect()

                # add all restrictions
                ## none of the non_ascii tokens.
                if not allow_non_ascii:
                    logits_softmaxed_currpos[self.prompts[0]._nonascii_toks.to(main_device)] = -torch.inf

                ## also do not allow the lst used token
                logits_softmaxed_currpos[control_toks[pos-1]] = -torch.inf

                cum_logits += logits_softmaxed_currpos
                logits_softmaxed_cached.append(logits_softmaxed_currpos)

                if top_k is not None:
                    topk = top_k
                    lm_topks.append(logits_softmaxed_currpos.topk(topk, dim=0).indices)
                else:
                    lm_topks.append(find_top_p(logits_softmaxed_currpos, top_p)[0])  # first element is the indices

                if not accumulate:
                    break

            # we are using topk finall, forget top_p. Always if case gets activated
            if top_p is None:
                # use topk, handle zero intersection cases by increasing topk
                # random sampling done to take a subset of samples to get some valid token proposals

                while True:
                    # random_indices = random.sample(range(len(lm_topks)), intersection_across_examples)
                    # intersection = set(lm_topks[random_indices[0]].tolist())
                    # for i in random_indices[1:]:
                    #     intersection = intersection.intersection(lm_topks[i].tolist())
                    all_intersections = []
                    for _ in range(num_intersections):
                        random_indices = random.sample(range(len(lm_topks)), intersection_across_examples)
                        intersection = set(lm_topks[random_indices[0]].tolist())
                        for i in random_indices[1:]:
                            intersection = intersection.intersection(lm_topks[i].tolist())
                        all_intersections.append(intersection)

                    intersection = set().union(*all_intersections)

                    top_indices = torch.tensor(list(intersection), device=main_device)
                    print(f"Intersection set size: {len(top_indices)}")
                    print(f"Candidate tokens: {workers[0].tokenizer.decode(top_indices)}")
                    if len(top_indices) >= (2 * minimum_keep_for_selection):
                        topk = top_k
                        break

                    else:
                        if top_k is not None:
                            topk += 2
                            print(f">>>> Topk increased to {topk} <<<<")
                            lm_topks = []
                            for logits_softmaxed_currpos in logits_softmaxed_cached:
                                lm_topks.append(logits_softmaxed_currpos.topk(topk, dim=0).indices)

            else:
                cum_logits[cum_logits == -torch.inf] = -9e11
                cum_logits = F.log_softmax(cum_logits, dim=0)
                top_indices = find_top_p(cum_logits, top_p)[0]

            # Stage 2 End: top_indices contains the top k token proposals for the current position

            # Stage 3: Calculate gradients for the current position

            c_token = control_toks[pos]
            if c_token not in top_indices:
                # valid_tokens.append(current_token)
                top_indices = torch.cat([top_indices, torch.tensor([c_token], device=main_device)], dim=0)

            del logits_softmaxed_currpos, logits_softmaxed_cached; gc.collect()
            torch.cuda.empty_cache()

            grad = self.get_grads(main_device, pos, top_indices)

            # from the gradient array, select the current position, and only keep the values of the top_indices
            #grad_currpos = grad[pos, :]
            #grad_currpos = grad[0]

            # new implementation- SELECTION STEP
            selection_min_loss = 9e9
            selections = minimum_keep_for_selection
            lowest_selection = None
            topk_indices = top_indices[(-grad).topk(selections, dim=-1).indices[0]]
            #topk_indices = top_indices[(-grad_currpos).topk(selections, dim=0).indices]

            # Stage 4: Selection Step
            # mask = torch.full((len(grad_currpos),), float('inf'), dtype=grad_currpos.dtype, device=grad_currpos.device)
            # mask[top_indices] = grad_currpos[top_indices]
            #

            # lowest_selection = None
            # selections = minimum_keep_for_selection
            # topk_indices = (-mask).topk(selections, dim=0).indices

            initial_selection = copy.deepcopy(control_toks[pos])
            #logits, target_slice_starts, targets_ids_list = [], [], []
            for choice in topk_indices:

                control_toks[pos] = choice
                loss_new = 0


                current_prompt_choice = workers[0].tokenizer.decode(control_toks)

                if current_prompt_choice in self.loss_cache.keys():
                    loss_new = self.loss_cache[current_prompt_choice]


                else:
                    record1, record2, record3 = None, None, None
                    percentage = 0.5

                    for iter_idx in range(0, round(1//percentage + 0.001)):
                        logits, target_slice_starts, targets_ids_list, _prompts_obj_list, padding_cnts = self.prompts[0].logits_batched_gen(
                            self.test_workers[0].model, control_toks, percentage=0.5, iter_index=iter_idx)

                        logits, target_slice_starts, targets_ids_list = [logits], [target_slice_starts], [targets_ids_list]

                        # loss_new, mistakes_perc = self.prompts[k].selection_loss(logits[0], target_slice_starts[0],
                        #                                                 targets_ids_list[0]).mean()

                        loss_new, mist_perc, control_loss = self.prompts[k].selection_loss(logits[0], target_slice_starts[0],
                                                                  targets_ids_list[0], _prompts_obj_list, padding_cnts)

                        if record1 is None:
                            record1, record2, record3 = loss_new, mist_perc, control_loss
                        else:
                            record1, record2, record3 = torch.cat((record1, loss_new), dim=0), torch.cat((record2, mist_perc), dim=0), torch.cat((record3, control_loss), dim=0)

                        torch.cuda.empty_cache()

                    loss_new, mist_perc, control_loss = record1.mean(), record2.mean(), record3.mean()
                    #loss_new, mist_perc, control_loss = loss_new.mean(), mist_perc.mean(), control_loss.mean()

                    self.loss_cache[current_prompt_choice] = loss_new + control_weight * control_loss

                # print(f">Prompt Choice: {current_prompt_choice} Loss: {loss_new}, Mistakes: {mist_perc}, Control Loss: {control_loss}, Total Loss: {loss_new + (control_weight) * control_loss}")
                print(
                    f">Prompt Choice: {current_prompt_choice} Loss: {loss_new}, Mistakes: {mist_perc}, Control Loss: {control_loss}, Total Loss: {mist_perc + (control_weight / 10) * control_loss}")

                #loss_new = loss_new + control_weight * control_loss
                loss_new = mist_perc + (control_weight/10) * control_loss

                # print(loss_new)

                if loss_new <= selection_min_loss:
                    selection_min_loss = loss_new
                    lowest_selection = choice

                if ((choice == ending_token or ("." in self.workers[0].tokenizer.decode([choice]))) and pos >= 0.5 * self.initial_length):
                    selection_min_loss = loss_new
                    lowest_selection = choice
                    break

            control_toks[pos] = lowest_selection

        if sequentially_increasing:
            if lowest_selection == initial_selection or lowest_selection != initial_selection:
                print("No improvement in the current position. Moving to the next position.")
                # we reached local optima for the current position
                self.prompts[0].current_pos = (self.prompts[0].current_pos + 1)

                self.loss_cache = {}
                # increase the control length with ellipsis (...)

                # if ((lowest_selection == ending_token or ("." in self.workers[0].tokenizer.decode([lowest_selection]))) and self.prompts[0].current_pos >= 0.75 * control_len):
                #     self.prompts[0].current_pos = (self.prompts[0].current_pos) % control_len



                # if (lowest_selection == ending_token or ("." in self.workers[0].tokenizer.decode([lowest_selection]))):
                #     ending_token = self.test_workers[0].tokenizer.encode("sample.")[-1]

                if self.prompts[0].current_pos >= control_len:
                    # if lowest_selection == ending_token:
                    #     self.prompts[0].current_pos = (self.prompts[0].current_pos) % control_len
                    if (lowest_selection == ending_token or ("." in self.workers[0].tokenizer.decode([lowest_selection]))):
                        self.prompts[0].current_pos = (self.prompts[0].current_pos) % control_len
                    else:
                        control_toks = torch.cat(
                            (control_toks, torch.tensor([ellipsis_token], device=control_toks.device).view(1)), dim=0)
                        control_len += 1
                # self.loss_cache = {}

            elif lowest_selection.item() in self.selection_history.keys():
                # circular, we are selecting the same token again.
                # choose the candidate with the lowest loss in the dictionary
                # select the one in history with the lowest loss
                minval, l = 9e9, None
                for (key, value) in self.selection_history.items():
                    if value < minval:
                        minval = value
                        l = key
                control_toks[pos] = l
                selection_min_loss = minval
                self.prompts[0].current_pos = (self.prompts[0].current_pos + 1)
                self.loss_cache = {}
                self.selection_history = {}
                print(
                    "<--Circular selection. Selecting the candidate with the lowest loss in the history. Moving to next position.-->")

            else:
                print("<--Updated in the same position. Reevaluating further changes..-->")
                self.selection_history[lowest_selection.item()] = selection_min_loss


        else:
            if lowest_selection == initial_selection or lowest_selection != initial_selection:
                print("<--No improvement in the current position. Moving to the next position.-->")
                self.prompts[0].current_pos = (self.prompts[0].current_pos + 1) % control_len
                self.loss_cache = {}
            elif lowest_selection.item() in self.selection_history.keys():
                # circular, we are selecting the same token again.
                # choose the candidate with the lowest loss in the dictionary
                # select the one in history with the lowest loss
                minval, l = 9e9, None
                for (key, value) in self.selection_history.items():
                    if value < minval:
                        minval = value
                        l = key
                control_toks[pos] = l
                selection_min_loss = minval
                self.prompts[0].current_pos = (self.prompts[0].current_pos + 1) % control_len
                self.loss_cache = {}
                self.selection_history = {}
                print(
                    "<--Circular selection. Selecting the candidate with the lowest loss in the history. Moving to next position.-->")
            else:
                print("<--Updated in the same position. Reevaluating further changes.-->")
                self.selection_history[lowest_selection.item()] = selection_min_loss

        del lm_probs, our_logit, top_indices, grad, logits, ids, target_slice_starts, targets_ids_list;
        gc.collect()
        torch.cuda.empty_cache()

        decoded_str = workers[0].tokenizer.decode(control_toks, skip_special_tokens=True)
        print(f"New Control: {decoded_str}, New Loss: {selection_min_loss}")
        return control_toks, decoded_str, selection_min_loss

    def step(self,
             batch_size=1024,
             topk=256,
             temp=1,
             topq=5,
             allow_non_ascii=True,
             target_weight=1,
             control_weight=0.2,
             verbose=False,
             opt_only=False,
             filter_cand=True):

        # GCG currently does not support optimization_only mode,
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        # for j, worker in enumerate(self.workers):
        #     worker(self.prompts[j], "grad", worker.model)

        # ****either this, our new implementation
        with torch.no_grad():
            _, next_control, cand_loss = self.morph_control(self.workers, top_p=None, top_k=topk, num_intersections=topq ,accumulate=True)
            # del grad;
            gc.collect()

        # ****or this, the original implementation
        # with torch.no_grad():
        #     control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
        #     control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        # del grad, control_cand ; gc.collect()
        #
        # # Search
        # loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        # with torch.no_grad():
        #     for j, cand in enumerate(control_cands):
        #         # Looping through the prompts at this level is less elegant, but
        #         # we can manage VRAM better this way
        #         progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
        #         for i in progress:
        #             for k, worker in enumerate(self.workers):
        #                 worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
        #             logits, ids = zip(*[worker.results.get() for worker in self.workers])
        #             loss[j*batch_size:(j+1)*batch_size] += sum([
        #                 target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device)
        #                 for k, (logit, id) in enumerate(zip(logits, ids))
        #             ])
        #             if control_weight != 0:
        #                 loss[j*batch_size:(j+1)*batch_size] += sum([
        #                     control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
        #                     for k, (logit, id) in enumerate(zip(logits, ids))
        #                 ])
        #             del logits, ids ; gc.collect()
        #
        #             if verbose:
        #                 progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
        #
        #     min_idx = loss.argmin()
        #     model_idx = min_idx // batch_size
        #     batch_idx = min_idx % batch_size
        #     next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        #
        # del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item()
        # return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)