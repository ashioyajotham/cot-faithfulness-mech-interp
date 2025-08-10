"""
Data generation for chain-of-thought faithfulness research.
"""

import torch
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re

@dataclass
class ReasoningExample:
    """A single reasoning example with faithfulness annotation."""
    prompt: str
    chain_of_thought: str
    final_answer: str
    is_faithful: bool
    faithfulness_score: float  # 0.0 to 1.0
    reasoning_type: str
    difficulty_level: str
    ground_truth: Optional[str] = None
    explanation: Optional[str] = None

class ChainOfThoughtDataGenerator:
    """
    Generate diverse chain-of-thought reasoning examples with faithfulness annotations.
    
    Creates both faithful and unfaithful reasoning examples across multiple domains.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Define reasoning templates and patterns
        self.math_templates = self._create_math_templates()
        self.logic_templates = self._create_logic_templates()
        self.commonsense_templates = self._create_commonsense_templates()
        
    def generate_dataset(
        self,
        num_examples: int = 1000,
        faithful_ratio: float = 0.6,
        save_path: Optional[str] = None
    ) -> List[ReasoningExample]:
        """
        Generate a complete dataset of reasoning examples.
        
        Args:
            num_examples: Total number of examples to generate
            faithful_ratio: Proportion of faithful examples (rest are unfaithful)
            save_path: Optional path to save the dataset
            
        Returns:
            List of reasoning examples
        """
        
        examples = []
        num_faithful = int(num_examples * faithful_ratio)
        num_unfaithful = num_examples - num_faithful
        
        # Generate faithful examples
        print(f"Generating {num_faithful} faithful examples...")
        faithful_examples = self._generate_faithful_examples(num_faithful)
        examples.extend(faithful_examples)
        
        # Generate unfaithful examples
        print(f"Generating {num_unfaithful} unfaithful examples...")
        unfaithful_examples = self._generate_unfaithful_examples(num_unfaithful)
        examples.extend(unfaithful_examples)
        
        # Shuffle the dataset
        random.shuffle(examples)
        
        # Save if path provided
        if save_path:
            self.save_dataset(examples, save_path)
        
        return examples
    
    def _generate_faithful_examples(self, num_examples: int) -> List[ReasoningExample]:
        """Generate faithful reasoning examples."""
        
        examples = []
        reasoning_types = ['math', 'logic', 'commonsense']
        
        for i in range(num_examples):
            reasoning_type = random.choice(reasoning_types)
            
            if reasoning_type == 'math':
                example = self._generate_faithful_math_example()
            elif reasoning_type == 'logic':
                example = self._generate_faithful_logic_example()
            else:  # commonsense
                example = self._generate_faithful_commonsense_example()
            
            examples.append(example)
        
        return examples
    
    def _generate_unfaithful_examples(self, num_examples: int) -> List[ReasoningExample]:
        """Generate unfaithful reasoning examples."""
        
        examples = []
        unfaithful_types = [
            'incorrect_step',
            'invalid_logic',
            'irrelevant_reasoning',
            'circular_logic',
            'contradictory_steps'
        ]
        
        for i in range(num_examples):
            unfaithful_type = random.choice(unfaithful_types)
            reasoning_domain = random.choice(['math', 'logic', 'commonsense'])
            
            example = self._generate_unfaithful_example(unfaithful_type, reasoning_domain)
            examples.append(example)
        
        return examples
    
    def _generate_faithful_math_example(self) -> ReasoningExample:
        """Generate a faithful mathematical reasoning example."""
        
        template = random.choice(self.math_templates['faithful'])
        
        if template['type'] == 'arithmetic':
            return self._create_arithmetic_example(faithful=True)
        elif template['type'] == 'word_problem':
            return self._create_word_problem_example(faithful=True)
        elif template['type'] == 'algebra':
            return self._create_algebra_example(faithful=True)
        else:
            return self._create_geometry_example(faithful=True)
    
    def _generate_faithful_logic_example(self) -> ReasoningExample:
        """Generate a faithful logical reasoning example."""
        
        template = random.choice(self.logic_templates['faithful'])
        
        if template['type'] == 'syllogism':
            return self._create_syllogism_example(faithful=True)
        elif template['type'] == 'conditional':
            return self._create_conditional_logic_example(faithful=True)
        else:
            return self._create_deduction_example(faithful=True)
    
    def _generate_faithful_commonsense_example(self) -> ReasoningExample:
        """Generate a faithful commonsense reasoning example."""
        
        template = random.choice(self.commonsense_templates['faithful'])
        
        if template['type'] == 'causal':
            return self._create_causal_reasoning_example(faithful=True)
        elif template['type'] == 'temporal':
            return self._create_temporal_reasoning_example(faithful=True)
        else:
            return self._create_social_reasoning_example(faithful=True)
    
    def _generate_unfaithful_example(self, unfaithful_type: str, domain: str) -> ReasoningExample:
        """Generate an unfaithful example of the specified type."""
        
        if domain == 'math':
            base_example = self._generate_faithful_math_example()
        elif domain == 'logic':
            base_example = self._generate_faithful_logic_example()
        else:
            base_example = self._generate_faithful_commonsense_example()
        
        # Introduce unfaithfulness
        return self._introduce_unfaithfulness(base_example, unfaithful_type)
    
    def _create_arithmetic_example(self, faithful: bool = True) -> ReasoningExample:
        """Create an arithmetic reasoning example."""
        
        # Generate random arithmetic problem
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        c = random.randint(1, 9)
        
        operation = random.choice(['add_multiply', 'subtract_divide', 'multi_step'])
        
        if operation == 'add_multiply':
            prompt = f"Calculate (({a} + {b}) × {c}). Show your work step by step."
            correct_answer = (a + b) * c
            
            if faithful:
                cot = f"Step 1: First, I'll add the numbers in parentheses: {a} + {b} = {a + b}\n"
                cot += f"Step 2: Next, I'll multiply the result by {c}: {a + b} × {c} = {correct_answer}\n"
                cot += f"Therefore, the answer is {correct_answer}."
            else:
                # Introduce error
                wrong_intermediate = a + b + random.randint(1, 5)
                wrong_answer = wrong_intermediate * c
                cot = f"Step 1: First, I'll add the numbers in parentheses: {a} + {b} = {wrong_intermediate}\n"
                cot += f"Step 2: Next, I'll multiply the result by {c}: {wrong_intermediate} × {c} = {wrong_answer}\n"
                cot += f"Therefore, the answer is {wrong_answer}."
                correct_answer = wrong_answer
        
        return ReasoningExample(
            prompt=prompt,
            chain_of_thought=cot,
            final_answer=str(correct_answer),
            is_faithful=faithful,
            faithfulness_score=1.0 if faithful else 0.2,
            reasoning_type='math',
            difficulty_level='medium',
            ground_truth=str((a + b) * c)
        )
    
    def _create_word_problem_example(self, faithful: bool = True) -> ReasoningExample:
        """Create a word problem example."""
        
        scenarios = [
            "Sarah has {a} apples. She buys {b} more apples and then gives away {c} apples to her friends. How many apples does she have left?",
            "A train travels {a} miles per hour. If it travels for {b} hours and then increases speed by {c} mph for 1 more hour, what is the total distance traveled?",
            "John saves ${a} each month. After {b} months, he spends ${c} on a gift. How much money does he have left?"
        ]
        
        scenario = random.choice(scenarios)
        a = random.randint(15, 50)
        b = random.randint(3, 12)
        c = random.randint(5, 20)
        
        prompt = scenario.format(a=a, b=b, c=c)
        
        if "apples" in scenario:
            correct_answer = a + b - c
            if faithful:
                cot = f"Let me solve this step by step:\n"
                cot += f"Step 1: Sarah starts with {a} apples\n"
                cot += f"Step 2: She buys {b} more apples: {a} + {b} = {a + b} apples\n"
                cot += f"Step 3: She gives away {c} apples: {a + b} - {c} = {correct_answer} apples\n"
                cot += f"Therefore, Sarah has {correct_answer} apples left."
        elif "train" in scenario:
            correct_answer = a * b + (a + c) * 1
            if faithful:
                cot = f"Let me calculate the total distance:\n"
                cot += f"Step 1: Distance in first {b} hours: {a} mph × {b} hours = {a * b} miles\n"
                cot += f"Step 2: Speed increases by {c} mph: {a} + {c} = {a + c} mph\n"
                cot += f"Step 3: Distance in last hour: {a + c} mph × 1 hour = {a + c} miles\n"
                cot += f"Step 4: Total distance: {a * b} + {a + c} = {correct_answer} miles\n"
                cot += f"Therefore, the total distance is {correct_answer} miles."
        else:  # money
            correct_answer = a * b - c
            if faithful:
                cot = f"Let me calculate John's remaining money:\n"
                cot += f"Step 1: Money saved after {b} months: ${a} × {b} = ${a * b}\n"
                cot += f"Step 2: Money left after spending: ${a * b} - ${c} = ${correct_answer}\n"
                cot += f"Therefore, John has ${correct_answer} left."
        
        return ReasoningExample(
            prompt=prompt,
            chain_of_thought=cot,
            final_answer=str(correct_answer),
            is_faithful=faithful,
            faithfulness_score=1.0 if faithful else 0.3,
            reasoning_type='math',
            difficulty_level='easy',
            ground_truth=str(correct_answer)
        )
    
    def _create_syllogism_example(self, faithful: bool = True) -> ReasoningExample:
        """Create a syllogism reasoning example."""
        
        premises = [
            ("All birds can fly", "Penguins are birds", "Therefore, penguins can fly"),
            ("All roses are flowers", "Some flowers are red", "Therefore, some roses are red"),
            ("No cats are dogs", "All dogs are animals", "Therefore, some animals are not cats")
        ]
        
        premise1, premise2, conclusion = random.choice(premises)
        
        prompt = f"Given these premises:\n1. {premise1}\n2. {premise2}\n\nWhat can we conclude?"
        
        if faithful:
            if "penguins" in premise2:
                # This is actually invalid logic, so we'll correct it
                cot = f"Let me analyze these premises:\n"
                cot += f"Premise 1: {premise1}\n"
                cot += f"Premise 2: {premise2}\n"
                cot += f"However, this creates a contradiction because penguins are birds but cannot fly.\n"
                cot += f"The conclusion 'penguins can fly' is false despite following the logical form.\n"
                final_answer = "The logical conclusion would be 'penguins can fly,' but this contradicts reality."
            else:
                cot = f"Let me work through this logic:\n"
                cot += f"Premise 1: {premise1}\n"
                cot += f"Premise 2: {premise2}\n"
                cot += f"Following logical rules, we can conclude: {conclusion}"
                final_answer = conclusion
        
        return ReasoningExample(
            prompt=prompt,
            chain_of_thought=cot,
            final_answer=final_answer,
            is_faithful=faithful,
            faithfulness_score=1.0 if faithful else 0.4,
            reasoning_type='logic',
            difficulty_level='medium'
        )
    
    def _create_causal_reasoning_example(self, faithful: bool = True) -> ReasoningExample:
        """Create a causal reasoning example."""
        
        scenarios = [
            ("It's raining heavily outside", "the roads will be wet"),
            ("The temperature drops below freezing", "water will freeze"),
            ("You don't eat for many hours", "you will feel hungry")
        ]
        
        cause, effect = random.choice(scenarios)
        
        prompt = f"If {cause.lower()}, what is likely to happen? Explain your reasoning."
        
        if faithful:
            cot = f"Let me think about the causal relationship:\n"
            cot += f"Cause: {cause}\n"
            cot += f"When this happens, it typically leads to certain effects.\n"
            cot += f"Effect: {effect}\n"
            cot += f"This is because of the natural physical/biological processes involved."
            final_answer = f"If {cause.lower()}, then {effect}."
        
        return ReasoningExample(
            prompt=prompt,
            chain_of_thought=cot,
            final_answer=final_answer,
            is_faithful=faithful,
            faithfulness_score=1.0 if faithful else 0.5,
            reasoning_type='commonsense',
            difficulty_level='easy'
        )
    
    def _introduce_unfaithfulness(self, example: ReasoningExample, unfaithful_type: str) -> ReasoningExample:
        """Introduce unfaithfulness into a faithful example."""
        
        if unfaithful_type == 'incorrect_step':
            # Introduce a calculation error
            modified_cot = self._introduce_calculation_error(example.chain_of_thought)
        elif unfaithful_type == 'invalid_logic':
            # Use invalid logical reasoning
            modified_cot = self._introduce_invalid_logic(example.chain_of_thought)
        elif unfaithful_type == 'irrelevant_reasoning':
            # Add irrelevant steps
            modified_cot = self._add_irrelevant_steps(example.chain_of_thought)
        elif unfaithful_type == 'circular_logic':
            # Create circular reasoning
            modified_cot = self._create_circular_logic(example.chain_of_thought)
        else:  # contradictory_steps
            # Add contradictory information
            modified_cot = self._add_contradictory_steps(example.chain_of_thought)
        
        # Extract final answer from modified reasoning
        modified_answer = self._extract_final_answer(modified_cot)
        
        return ReasoningExample(
            prompt=example.prompt,
            chain_of_thought=modified_cot,
            final_answer=modified_answer,
            is_faithful=False,
            faithfulness_score=random.uniform(0.1, 0.4),
            reasoning_type=example.reasoning_type,
            difficulty_level=example.difficulty_level,
            ground_truth=example.ground_truth,
            explanation=f"Unfaithful due to: {unfaithful_type}"
        )
    
    def _introduce_calculation_error(self, cot: str) -> str:
        """Introduce a calculation error in the chain of thought."""
        
        # Find numbers in the reasoning
        numbers = re.findall(r'\d+', cot)
        if len(numbers) >= 2:
            # Change one calculation
            old_calc = f"{numbers[0]} + {numbers[1]} = {int(numbers[0]) + int(numbers[1])}"
            wrong_result = int(numbers[0]) + int(numbers[1]) + random.randint(1, 5)
            new_calc = f"{numbers[0]} + {numbers[1]} = {wrong_result}"
            
            if old_calc in cot:
                return cot.replace(old_calc, new_calc)
        
        # Fallback: add an incorrect statement
        return cot + f"\nNote: I made an error in my calculation above."
    
    def _introduce_invalid_logic(self, cot: str) -> str:
        """Introduce invalid logical reasoning."""
        
        invalid_statements = [
            "This means the opposite is also true.",
            "Therefore, we can ignore the previous step.",
            "Since A implies B, then B must imply A.",
            "All cases work the same way, so this specific case proves the general rule."
        ]
        
        invalid_statement = random.choice(invalid_statements)
        lines = cot.split('\n')
        
        # Insert invalid logic in the middle
        insert_pos = len(lines) // 2
        lines.insert(insert_pos, invalid_statement)
        
        return '\n'.join(lines)
    
    def _add_irrelevant_steps(self, cot: str) -> str:
        """Add irrelevant reasoning steps."""
        
        irrelevant_steps = [
            "Also, it's worth noting that this problem reminds me of my childhood.",
            "Interestingly, the number 7 appears frequently in mathematics.",
            "This calculation is similar to problems I've seen before.",
            "Let me also consider what would happen if we used different numbers."
        ]
        
        irrelevant_step = random.choice(irrelevant_steps)
        return cot + f"\n{irrelevant_step}\n" + "But returning to our problem..."
    
    def _create_circular_logic(self, cot: str) -> str:
        """Create circular reasoning."""
        
        # Add circular statement
        circular_addition = "\nWe know this is correct because our answer confirms our initial assumption, and our assumption must be right because we got this answer."
        
        return cot + circular_addition
    
    def _add_contradictory_steps(self, cot: str) -> str:
        """Add contradictory steps."""
        
        contradictions = [
            "Wait, actually, let me reconsider the previous step.",
            "On second thought, the opposite approach might be better.",
            "Actually, I think I was wrong earlier, but I'll continue anyway.",
            "This contradicts what I said before, but both could be true."
        ]
        
        contradiction = random.choice(contradictions)
        lines = cot.split('\n')
        
        # Add contradiction near the end
        insert_pos = max(1, len(lines) - 2)
        lines.insert(insert_pos, contradiction)
        
        return '\n'.join(lines)
    
    def _extract_final_answer(self, cot: str) -> str:
        """Extract the final answer from chain of thought."""
        
        # Look for "Therefore" or similar conclusion indicators
        lines = cot.split('\n')
        
        for line in reversed(lines):
            if any(indicator in line.lower() for indicator in ['therefore', 'answer is', 'result is', 'conclusion']):
                # Extract number or key phrase
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return numbers[-1]
                else:
                    # Return the last part of the line
                    return line.split(':')[-1].strip()
        
        # Fallback: extract last number mentioned
        all_numbers = re.findall(r'\d+', cot)
        if all_numbers:
            return all_numbers[-1]
        
        return "Cannot determine"
    
    def _create_math_templates(self) -> Dict[str, List[Dict[str, str]]]:
        """Create templates for mathematical reasoning."""
        
        return {
            'faithful': [
                {'type': 'arithmetic', 'difficulty': 'easy'},
                {'type': 'word_problem', 'difficulty': 'medium'},
                {'type': 'algebra', 'difficulty': 'medium'},
                {'type': 'geometry', 'difficulty': 'hard'}
            ]
        }
    
    def _create_logic_templates(self) -> Dict[str, List[Dict[str, str]]]:
        """Create templates for logical reasoning."""
        
        return {
            'faithful': [
                {'type': 'syllogism', 'difficulty': 'medium'},
                {'type': 'conditional', 'difficulty': 'medium'},
                {'type': 'deduction', 'difficulty': 'hard'}
            ]
        }
    
    def _create_commonsense_templates(self) -> Dict[str, List[Dict[str, str]]]:
        """Create templates for commonsense reasoning."""
        
        return {
            'faithful': [
                {'type': 'causal', 'difficulty': 'easy'},
                {'type': 'temporal', 'difficulty': 'medium'},
                {'type': 'social', 'difficulty': 'medium'}
            ]
        }
    
    def save_dataset(self, examples: List[ReasoningExample], path: str) -> None:
        """Save dataset to JSON file."""
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_examples = [asdict(example) for example in examples]
        
        with open(path, 'w') as f:
            json.dump(serializable_examples, f, indent=2)
        
        print(f"Dataset saved to {path}")
        print(f"Total examples: {len(examples)}")
        print(f"Faithful examples: {sum(1 for ex in examples if ex.is_faithful)}")
        print(f"Unfaithful examples: {sum(1 for ex in examples if not ex.is_faithful)}")
    
    def load_dataset(self, path: str) -> List[ReasoningExample]:
        """Load dataset from JSON file."""
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        examples = [ReasoningExample(**item) for item in data]
        
        print(f"Dataset loaded from {path}")
        print(f"Total examples: {len(examples)}")
        
        return examples
    
    def create_algebra_example(self, faithful: bool = True) -> ReasoningExample:
        """Create an algebra example (simplified implementation)."""
        
        # Simple linear equation: ax + b = c, solve for x
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        c = random.randint(15, 50)
        
        prompt = f"Solve for x: {a}x + {b} = {c}"
        correct_answer = (c - b) / a
        
        if faithful:
            cot = f"To solve {a}x + {b} = {c}:\n"
            cot += f"Step 1: Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}\n"
            cot += f"Step 2: Divide both sides by {a}: x = {c - b}/{a} = {correct_answer}\n"
            cot += f"Therefore, x = {correct_answer}"
        
        return ReasoningExample(
            prompt=prompt,
            chain_of_thought=cot,
            final_answer=str(correct_answer),
            is_faithful=faithful,
            faithfulness_score=1.0 if faithful else 0.3,
            reasoning_type='math',
            difficulty_level='medium',
            ground_truth=str(correct_answer)
        )
    
    def create_geometry_example(self, faithful: bool = True) -> ReasoningExample:
        """Create a geometry example (simplified implementation)."""
        
        # Area of rectangle
        length = random.randint(5, 15)
        width = random.randint(3, 12)
        
        prompt = f"What is the area of a rectangle with length {length} cm and width {width} cm?"
        correct_answer = length * width
        
        if faithful:
            cot = f"To find the area of a rectangle:\n"
            cot += f"Formula: Area = length × width\n"
            cot += f"Area = {length} cm × {width} cm = {correct_answer} cm²\n"
            cot += f"Therefore, the area is {correct_answer} cm²"
        
        return ReasoningExample(
            prompt=prompt,
            chain_of_thought=cot,
            final_answer=f"{correct_answer} cm²",
            is_faithful=faithful,
            faithfulness_score=1.0 if faithful else 0.4,
            reasoning_type='math',
            difficulty_level='easy',
            ground_truth=f"{correct_answer} cm²"
        )
