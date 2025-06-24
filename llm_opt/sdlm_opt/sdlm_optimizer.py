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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdlm_model = None
        self.sdlm_variable = None
        
    def init_sdlm(self, model, tokenizer):
        """Initialize the SDLM model and variable."""
        self.sdlm_model = STGSDiffModel(
            model=model,
            tokenizer=tokenizer,
            temperature=1.0,
            stgs_kwargs={
                "stgs_hard": False,
                "init_temperature": 1.0,
                "learnable_temperature": True,
                "hidden_state_conditioning": True,
            },
            device=model.device
        )
        
        # Initialize the prompt variable
        self.sdlm_variable = Variable(
            initial_string=self.control_str(),
            tokenizer=tokenizer,
            temperature=1.0,
            learnable_temperature=True,
            device=model.device
        )
    
    def grad(self, model, current_pos, valid_tokens, **kwargs):
        """
        Compute gradients using SDLM's differentiable text generation.
        
        Args:
            model: The target model
            current_pos: Current position in the prompt
            valid_tokens: List of valid token indices
            
        Returns:
            Gradients for the current position
        """
        if self.sdlm_model is None:
            self.init_sdlm(model, self.tokenizer)
        
        # Get the current control tokens
        control_tokens = self.control_toks
        
        # Create input for SDLM
        input_ids = torch.tensor([control_tokens], device=model.device)
        
        # Forward pass through SDLM
        with torch.enable_grad():
            outputs = self.sdlm_model(
                input_ids=input_ids,
                labels=input_ids,  # Use the same input as target for auto-regressive modeling
                output_hidden_states=True
            )
            
            # Compute loss (you can customize this based on your needs)
            loss = outputs.loss
            
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pos = 0
        
    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True, **kwargs):
        """
        Sample new controls based on gradients using SDLM.
        
        Args:
            grad: Gradient tensor
            batch_size: Number of candidates to generate
            topk: Top-k sampling parameter
            temp: Temperature for sampling
            allow_non_ascii: Whether to allow non-ASCII tokens
            
        Returns:
            List of candidate control tokens
        """
        # Get the current control tokens
        control_toks = self.control_toks
        
        # Update the current position
        self.current_pos = (self.current_pos + 1) % len(control_toks)
        
        # Get the current prompt's SDLM variable
        prompt = self._prompts[0]  # Assuming single prompt for simplicity
        if not hasattr(prompt, 'sdlm_variable') or prompt.sdlm_variable is None:
            return [control_toks]  # Return original if not initialized
            
        # Apply gradient update to the SDLM variable
        with torch.no_grad():
            # Update logits with gradient
            lr = kwargs.get('lr', 0.1)
            prompt.sdlm_variable.logits.data.add_(prompt.sdlm_variable._grad * lr)
            
            # Sample new tokens
            new_tokens = prompt.sdlm_variable.sample()
            
            # Convert to list of tokens
            new_control = new_tokens[0].tolist()
            
            return [new_control]


class SDLMMultiPrompter(BaseMultiPrompter):
    """
    A multi-prompter that uses SDLM for optimization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cache = {}
        self.initial_length = len(self.prompts[0].control_toks)
        
    def step(self, batch_size=1024, topk=256, temp=1, topq=5, allow_non_ascii=True,
             target_weight=1, control_weight=0.2, verbose=False, opt_only=False, filter_cand=True):
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
        main_device = self.prompts[0].device
        
        # Get gradients using SDLM
        grads = self.prompts[0].grad(
            self.models[0],
            self.prompts[0].current_pos,
            None,  # valid_tokens not needed for SDLM
            target_weight=target_weight,
            control_weight=control_weight
        )
        
        # Sample new controls
        control_cands = self.prompts[0].sample_control(
            grads,
            batch_size,
            topk=topk,
            temp=temp,
            allow_non_ascii=allow_non_ascii,
            lr=0.1  # Learning rate for SDLM updates
        )
        
        # Filter candidates if needed
        if filter_cand and not opt_only:
            control_cands = self.prompts[0].get_filtered_cands(
                0,  # worker_index
                control_cands,
                filter_cand=True,
                curr_control=self.control_str()
            )
        
        # Update control with the best candidate
        if control_cands:
            self.control_str = control_cands[0]
            
        # Return loss (placeholder)
        return 0.0
    
    def run(self, n_steps=100, batch_size=1024, topk=256, temp=1, topq=5, 
            target_weight=1, control_weight=0.2, anneal=True, test_steps=50,
            stop_on_success=True, verbose=True, filter_cand=True, **kwargs):
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
