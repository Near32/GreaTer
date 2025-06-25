import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import sys
import os

# Add SDLM to the path
#sdlm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'sdlm')
#sys.path.append(sdlm_path)
import sdlm
from sdlm.stgs_diff_model import STGSDiffModel
from sdlm.textgrad.variables import Variable
from sdlm.textgrad.optimizer import textgrad_optimize

from llm_opt.base.attack_manager import (
    Prompter as BasePrompter,
    PromptManager as BasePromptManager,
    MultiPrompter as BaseMultiPrompter,
    get_embeddings,
    get_embedding_matrix,
)

class SDLMPrompter(BasePrompter):
    """
    A prompter that uses SDLM for differentiable prompt optimization.
    """
    def __init__(
        self, 
        variable_init_temperature: float = 1.0,
        gradient_comp_batch_size: int = 1,
        stgs_kwargs: Dict[str, Any] = {
            "stgs_hard": False,
            "init_temperature": 1.0,
            "learnable_temperature": True,
            "hidden_state_conditioning": True,
        },
        *args, 
        **kwargs,
    ):
        """
        Initialize the SDLM prompter.

        Args:
            variable_init_temperature: Initial temperature for the SDLM variable
            gradient_comp_batch_size: Batch size for gradient computation
            stgs_kwargs: Keyword arguments for the STGS model
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.variable_init_temperature = variable_init_temperature
        self.gradient_comp_batch_size = gradient_comp_batch_size
        self.stgs_kwargs = stgs_kwargs
        self.sdlm_model = None
        self.sdlm_variable = None
        
    def init_sdlm(self, model, tokenizer):
        """Initialize the SDLM model and variable."""
        self.sdlm_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            stgs_kwargs=self.stgs_kwargs,
            device=model.device
        )
        
        # Initialize the prompt variable
        self.sdlm_variable = Variable(
            initial_string=self.control_str,
            tokenizer=tokenizer,
            temperature=self.variable_init_temperature,
            learnable_temperature=True,
            device=model.device
        )
    
    def grad(
        self, 
        model, 
        current_pos, 
        valid_tokens, 
        temperature=1, 
        control_weight=0.2,
        **kwargs
    ):
        """
        Compute gradients using SDLM's differentiable text generation.
        
        Args:
            model: The target model
            current_pos: Current position in the prompt
            valid_tokens: List of valid token indices
            temperature: Temperature for the focused loss computation
            control_weight: Weight for the control loss
        Returns:
            Gradients for the current position
        """
        if self.sdlm_model is None:
            self.init_sdlm(model, self.tokenizer)
            
        # Get the current control tokens
        control_tokens = self.control_toks
        
        # Create input for SDLM
        input_ids = self.input_ids.to(device=model.device)
        # Set SDLM variable to current input:
        self.sdlm_variable.reset(initial_ids=input_ids)
        input_one_hots = []
        for bidx in range(self.gradient_comp_batch_size):
            diff_input_ids, diff_one_hot, decoded_string = self.sdlm_variable.forward()
            input_one_hots.append(diff_one_hot)
        input_one_hots = torch.stack(input_one_hots)

        # Forward pass through SDLM
        with torch.enable_grad():
            outputs = self.sdlm_model(
                input_one_hots=input_one_hots,
                output_hidden_states=True
            )
            
            # Compute loss (you can customize this based on your needs)
            stgs_logits = outputs.stgs_logits
            #goals = self.input_ids[self._goal_slice]
            targets = input_ids[self._target_slice].repeat(self.gradient_comp_batch_size, 1)
            loss_crit = nn.CrossEntropyLoss(reduction='mean')
            import ipdb; ipdb.set_trace()
            loss = loss_crit(
                stgs_logits[:, self._loss_slice, :], 
                targets.detach(),
            )

            # Compute control loss, i.e. perplexity:
            control_output_slice = slice(self._control_slice.start - 1, self._control_slice.stop - 1)
            control_target_slice = slice(self._control_slice.start, self._control_slice.stop)
            control_targets = input_ids[control_target_slice].repeat(self.gradient_comp_batch_size, 1)
            control_loss = loss_crit(
                stgs_logits[:, control_output_slice, :], 
                control_targets.detach(),
            )

            if self._focused_target_slice:
                # loss computation requires shifted slices:
                focused_loss_slice = slice(self._focused_target_slice.start - 1, self._focused_target_slice.stop - 1)
                focused_targets = input_ids[self._focused_target_slice]
                focused_targets = focused_targets.repeat(self.gradient_comp_batch_size, 1)
                focused_loss = loss_crit(
                    stgs_logits[:, focused_loss_slice, :] / temperature, 
                    focused_targets.detach()
                )
                loss = focused_loss+control_weight*control_loss
            else:
                loss = loss+control_weight*control_loss
            
            # Backward pass
            loss.backward()
            
            # Get gradients from the SDLM variable
            grads = self.sdlm_variable.logits.grad
            
            # Only return gradients for the current position
            return grads[0, current_pos].cpu().numpy()


class SDLMPromptManager(BasePromptManager):
    """
    A prompt manager that uses SDLM for optimization.
    """
    def __init__(
        self, 
        #learning_rate: float = 0.1,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.1 #learning_rate
        self.current_pos = 0
    
    def grad(
        self, 
        model, 
        current_pos, 
        valid_tokens,
        control_weight=0.2,
        **kwargs,
    ):
        rsum = sum([
            prompt.grad(
                model, 
                current_pos, 
                valid_tokens,
                control_weight=control_weight,
                **kwargs,
            ) for prompt in self._prompts
        ])

        return rsum

    def sample_control(
        self, 
        grad, 
        batch_size, 
        topk=256, 
        temp=1, 
        allow_non_ascii=True, 
        **kwargs,
    ):
        """
        Sample new controls based on gradients using SDLM.
        
        Args:
            grad: Gradient tensor
            batch_size: Number of candidates to generate
            topk: Top-k sampling parameter
            temp: Temperature for sampling
            allow_non_ascii: Whether to allow non-ASCII tokens
            
        Returns:
            Tensor of shape (batch_size, seq_len)
        """
        current_pos = self.current_pos
        original_control_toks = self.control_toks.to(grad.device)
        current_control_toks = original_control_toks.repeat(batch_size, 1)

        # Apply gradient update to the SDLM variable
        with torch.no_grad():
            # Update logits with gradient
            lr = kwargs.get('lr', self.learning_rate)
            for prompt in self._prompts:
                prompt.sdlm_variable.logits.data.add_(prompt.sdlm_variable._grad * lr)
            
            # Sample new tokens
            new_tokens = [
                prompt.sdlm_variable.sample() 
                for prompt in self._prompts
            ]
            
            # Convert to list of tokens
            new_control = [
                new_token.tolist() 
                for new_token in new_tokens
            ]

        mixed_new_control_toks = [
            current_control_toks[i,:current_pos].tolist() + new_control[i][current_pos:] 
            for i in range(batch_size)
        ]

        self.current_pos = (self.current_pos + 1) % len(original_control_toks)
        return torch.tensor(mixed_new_control_toks, device=grad.device)


class SDLMMultiPrompter(BaseMultiPrompter):
    """
    A multi-prompter that uses SDLM for optimization.
    """
    def __init__(
        self, 
        #learning_rate: float = 0.1,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.1 #learning_rate
        self.loss_cache = {}
        self.initial_length = len(self.prompts[0].control_toks)
    
    def get_grads(
        self,
        main_device,
        current_pos,
        valid_tokens,
        control_weight=0.2,
    ):
        for j, worker in enumerate(self.workers):
            worker(
                self.prompts[j], 
                "grad", 
                worker.model, 
                current_pos, 
                valid_tokens,
                control_weight=control_weight,
            )

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            grad += new_grad

        return grad
    
    def step(
        self, 
        batch_size=1024, 
        topk=256, 
        temp=1, 
        topq=5, 
        allow_non_ascii=True,
        target_weight=1, 
        control_weight=0.2, 
        verbose=False, 
        opt_only=False, 
        filter_cand=True,
        *args,
        **kwargs,
    ):
        """
        Perform a single optimization step using SDLM.
        
        Args:
            batch_size: Number of candidates to generate
            topk: Top-k sampling parameter
            temp: Temperature for sampling
            topq: Top-q sampling parameter (unused in SDLM)
            allow_non_ascii: Whether to allow non-ASCII tokens
            target_weight: Weight for target loss
            control_weight: Weight for control loss
            verbose: Whether to print debug information
            opt_only: Whether to only optimize (no filtering)
            filter_cand: Whether to filter candidates
            
        Returns:
            Loss value
        """
        # Get the main device
        main_device = self.models[0].device
        
        # Get gradients using SDLM
        grads = self.get_grads(
            main_device=main_device,
            current_pos=self.prompts[0].current_pos,
            valid_tokens=None,  # valid_tokens not needed for SDLM
            control_weight=control_weight,
        )
        
        # Sample new controls
        control_cands = self.prompts[0].sample_control(
            grads,
            batch_size,
            topk=topk,
            temp=temp,
            allow_non_ascii=allow_non_ascii,
            lr=self.learning_rate  # Learning rate for SDLM updates
        )
        
        # Filter candidates if needed
        if filter_cand and not opt_only:
            control_cands = self.prompts[0].get_filtered_cands(
                0,  # worker_index
                control_cands,
                filter_cand=True,
                curr_control=self.control_str(temperature=1.0)
            )
        
        # Update control with the best candidate
        if control_cands:
            self.control_str = control_cands[0]

        next_control = self.control_str()
        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print('Current control:', next_control)
        return next_control, cand_loss.item()
    
    def deprecated_run(
        self, 
        n_steps=100, 
        batch_size=1024, 
        topk=256, 
        temp=1, 
        topq=5, 
        target_weight=1, 
        control_weight=0.2, 
        anneal=True, 
        test_steps=50,
        stop_on_success=True, 
        verbose=True, 
        filter_cand=True,
        *args,
        **kwargs,
    ):
        """
        Run the optimization process.
        
        Args:
            n_steps: Number of optimization steps
            batch_size: Batch size for optimization
            topk: Top-k sampling parameter
            temp: Temperature for sampling
            topq: Top-q sampling parameter
            target_weight: Weight for target loss
            control_weight: Weight for control loss
            anneal: Whether to use annealing
            test_steps: Number of steps between tests
            stop_on_success: Whether to stop on success
            verbose: Whether to print progress
            filter_cand: Whether to filter candidates
            **kwargs: Additional arguments
            
        Returns:
            Optimized control string
        """
        for step in range(n_steps):
            # Perform optimization step
            loss = self.step(
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                topq=topq,
                target_weight=target_weight,
                control_weight=control_weight,
                verbose=verbose,
                filter_cand=filter_cand
            )
            
            # Test the current solution periodically
            if step % test_steps == 0 or step == n_steps - 1:
                # Log progress
                if verbose:
                    print(f"Step {step}: Loss = {loss:.4f}")
                
                # Test on all workers
                self.test_all()
                
                # Check for early stopping
                if stop_on_success and all(self.test_results[-1]):
                    if verbose:
                        print("Stopping early - all tests passed!")
                    break
        
        return self.control_str
