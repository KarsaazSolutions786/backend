#!/usr/bin/env python3
"""
Bloom Response Generator
Generates natural, conversational responses that confirm processed intents and suggest habits.
"""

import json
import asyncio
import tempfile
import os
from typing import Dict, Any, Optional, List
from utils.logger import logger

class BloomResponseGenerator:
    """Generates natural responses using Bloom model with structured output."""
    
    def __init__(self, pytorch_env_path: str = "venv-pytorch/bin/python"):
        self.pytorch_env_path = pytorch_env_path
        self.model_path = "models/bloom-560m-8bit"
    
    async def generate_response(
        self,
        pipeline_output: Dict[str, Any],
        habit_suggestion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a natural response confirming processed intents.
        
        Args:
            pipeline_output: Complete pipeline output with database results
            habit_suggestion: Optional habit detection info {is_habit: bool, suggest: str}
            
        Returns:
            {
                "response": "<spoken reply>",
                "actions": [{"type": "note", "id": "<uuid>"}, ...],
                "follow_up": "<open-ended question>"
            }
        """
        try:
            # Check if we have any processed intents
            storage_results = pipeline_output.get("database_result", {}).get("storage_results", [])
            total_intents = pipeline_output.get("intent_result", {}).get("total_intents", 0)
            
            if total_intents == 0 or not storage_results:
                return self._generate_no_tasks_response()
            
            # Filter out low-confidence results
            confident_results = [
                r for r in storage_results 
                if r.get("confidence", 1.0) >= 0.15 and r.get("success", False)
            ]
            
            if not confident_results:
                return self._generate_no_tasks_response()
            
            # For now, use the enhanced detailed fallback response since it works better
            # than the current Bloom model outputs. We can enable Bloom later when it's optimized.
            logger.info("Using enhanced detailed fallback for better response quality")
            return self._generate_detailed_fallback_response(confident_results, habit_suggestion)
            
            # Bloom generation is commented out for now due to quality issues
            # TODO: Re-enable when Bloom model prompting is improved
            # system_prompt = self._build_system_prompt(confident_results, habit_suggestion)
            # response = await self._call_bloom_with_retry(system_prompt, confident_results, habit_suggestion)
            # return response
            
        except Exception as e:
            logger.error(f"Bloom response generation failed: {e}")
            return self._generate_detailed_fallback_response(storage_results, habit_suggestion)
    
    def _build_system_prompt(
        self, 
        storage_results: List[Dict[str, Any]], 
        habit_suggestion: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the system prompt for Bloom."""
        
        # Build confirmed items list
        confirmed_items = []
        actions = []
        
        for result in storage_results:
            intent = result.get("intent", "")
            data = result.get("data", {})
            record_id = result.get("record_id", "unknown")
            
            if intent == "create_note":
                content = data.get("content", data.get("title", "note"))
                confirmed_items.append(f'- Note: "{content}" (id {record_id[:8]}...) ✓')
                actions.append({"type": "note", "id": record_id})
                
            elif intent == "create_reminder":
                title = data.get("title", data.get("content", "reminder"))
                confirmed_items.append(f'- Reminder: "{title}" (id {record_id[:8]}...) ✓')
                actions.append({"type": "reminder", "id": record_id})
                
            elif intent == "create_ledger":
                amount = data.get("amount", 0)
                contact = data.get("contact_name", "contact")
                direction = data.get("direction", "owe")
                if direction == "owe":
                    confirmed_items.append(f'- Ledger: "${amount} with {contact}" (id {record_id[:8]}...) ✓')
                else:
                    confirmed_items.append(f'- Ledger: "${amount} from {contact}" (id {record_id[:8]}...) ✓')
                actions.append({"type": "ledger", "id": record_id})
        
        confirmed_text = "\n".join(confirmed_items)
        
        # Handle habit suggestion
        habit_text = "none"
        if habit_suggestion and habit_suggestion.get("is_habit", False):
            suggest_freq = habit_suggestion.get("suggest", "regularly")
            habit_text = f"User repeats this often → suggest making it {suggest_freq}."
        
        # Build system prompt with few-shot examples
        system_prompt = f"""User completed these tasks:
{confirmed_text}

Write a detailed confirmation response explaining what was accomplished and why it's helpful. Include specific details about each task.

Example: "Excellent! I've successfully saved your reminder to call mom tomorrow. This ensures you won't forget to connect with family and the reminder is now organized in your account with a timestamp for easy reference."

Detailed response:"""
        
        return system_prompt
    
    async def _call_bloom_with_retry(self, system_prompt: str, storage_results: List[Dict[str, Any]], habit_suggestion: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Bloom with retry logic for JSON parsing."""
        temperature = 0.5
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Adjust prompt for retries
                if attempt > 0:
                    prompt = f"REMINDER: Output valid JSON only.\n\n{system_prompt}"
                else:
                    prompt = system_prompt
                
                # Call Bloom
                result = await self._call_bloom_model(prompt, temperature, max_tokens=180)
                
                if result.get("success"):
                    response_text = result.get("generated_text", "").strip()
                    
                    # Instead of trying to parse JSON, treat this as a detailed text response
                    if response_text and len(response_text) > 10:
                        # Clean up the response
                        response_text = response_text.replace('\n', ' ').strip()
                        
                        # Filter out poor quality responses
                        if (response_text.count('[Hello]') > 3 or 
                            response_text.count('Hello') > 5 or
                            len(response_text.split()) < 5 or
                            response_text.count(response_text.split()[0]) > 5):
                            logger.warning(f"Poor quality Bloom response detected: {response_text[:100]}...")
                            # Skip this attempt and try again
                            continue
                        
                        # If it looks like a natural response, structure it
                        if not response_text.startswith("{"):
                            # Create the structured response from the natural text
                            structured_response = {
                                "response": response_text,
                                "actions": self._extract_actions_from_storage_results(storage_results),
                                "follow_up": self._generate_followup_from_response(response_text, habit_suggestion)
                            }
                            return structured_response
                
                # Reduce temperature for retry
                temperature = max(0.2, temperature - 0.1)
                
            except Exception as e:
                logger.error(f"Bloom call failed (attempt {attempt + 1}): {e}")
        
        # All retries failed - return fallback
        logger.warning("All Bloom retry attempts failed, using fallback")
        return self._generate_detailed_fallback_response(storage_results, habit_suggestion)
    
    async def _call_bloom_model(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Call the Bloom model using PyTorch environment."""
        try:
            # Check if PyTorch environment exists
            if not os.path.exists(self.pytorch_env_path):
                raise Exception("PyTorch environment not found")
            
            # Create temporary script for Bloom generation
            temp_script_content = f'''
import sys
import os
import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    start_time = time.time()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "{self.model_path}",
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("{self.model_path}", trust_remote_code=True)
    
    # Generate response
    prompt = """{prompt}"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens={max_tokens},
            temperature={temperature},
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new part (remove prompt)
    response = generated_text[len(prompt):].strip()
    
    processing_time = time.time() - start_time
    
    print(f"SUCCESS|{{response}}|{{processing_time}}")
    
except Exception as e:
    print(f"ERROR|{{str(e)}}|0.0")
'''
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(temp_script_content)
                temp_script_path = temp_file.name
            
            try:
                # Run the script using the PyTorch Python environment
                process = await asyncio.create_subprocess_exec(
                    self.pytorch_env_path, temp_script_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd="."
                )
                
                stdout, stderr = await process.communicate()
                
                # Parse output
                if process.returncode == 0 and stdout:
                    output = stdout.decode().strip()
                    if output.startswith("SUCCESS|"):
                        parts = output.split("|", 2)
                        if len(parts) >= 3:
                            generated_text = parts[1]
                            processing_time = float(parts[2])
                            
                            return {
                                "success": True,
                                "generated_text": generated_text,
                                "processing_time": processing_time
                            }
                
                # Handle errors
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                return {
                    "success": False,
                    "error": error_msg
                }
                
            finally:
                # Clean up temporary script
                try:
                    os.unlink(temp_script_path)
                except:
                    pass
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_actions_from_storage_results(self, storage_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract actions list from storage results."""
        actions = []
        for result in storage_results:
            intent = result.get("intent", "")
            record_id = result.get("record_id", "unknown")
            
            if intent == "create_note":
                actions.append({"type": "note", "id": record_id})
            elif intent == "create_reminder":
                actions.append({"type": "reminder", "id": record_id})
            elif intent == "create_ledger":
                actions.append({"type": "ledger", "id": record_id})
        
        return actions
    
    def _generate_followup_from_response(self, response_text: str, habit_suggestion: Optional[Dict[str, Any]]) -> str:
        """Generate an appropriate follow-up question based on the response and context."""
        response_lower = response_text.lower()
        
        # If habit suggestion is present and response doesn't already ask about it
        if (habit_suggestion and habit_suggestion.get("is_habit", False) and 
            "daily" not in response_lower and "reminder" not in response_lower):
            return f"Would you like me to set up a {habit_suggestion.get('suggest', 'regular')} reminder for this?"
        
        # Different follow-ups based on content
        if "note" in response_lower and "reminder" in response_lower:
            return "What other tasks or information would you like me to help you organize today?"
        elif "note" in response_lower:
            return "Is there anything else you'd like me to note or track for you?"
        elif "reminder" in response_lower:
            return "Are there any other reminders you'd like me to set up?"
        elif "ledger" in response_lower or "money" in response_lower or "$" in response_text:
            return "Do you have any other financial transactions you'd like me to record?"
        else:
            return "What else can I help you organize and manage today?"
    
    def _generate_detailed_fallback_response(self, storage_results: List[Dict[str, Any]], habit_suggestion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate concise but comprehensive fallback response."""
        if not storage_results:
            return self._generate_no_tasks_response()
        
        # Analyze what was accomplished
        notes = []
        reminders = []
        ledgers = []
        
        for result in storage_results:
            intent = result.get("intent", "")
            data = result.get("data", {})
            
            if intent == "create_note":
                content = data.get("content", data.get("title", "note"))
                notes.append(content)
            elif intent == "create_reminder":
                title = data.get("title", data.get("content", "reminder"))
                reminders.append(title)
            elif intent == "create_ledger":
                amount = data.get("amount", 0)
                contact = data.get("contact_name", "contact")
                direction = data.get("direction", "owe")
                ledgers.append({"amount": amount, "contact": contact, "direction": direction})
        
        # Build concise response
        response_parts = []
        
        # Opening
        if len(storage_results) == 1:
            response_parts.append("Perfect! I've completed your request.")
        else:
            response_parts.append(f"Excellent! I've completed {len(storage_results)} tasks for you.")
        
        # Detail each accomplishment concisely
        task_details = []
        
        for note in notes:
            task_details.append(f"saved your note '{note}'")
        
        for reminder in reminders:
            task_details.append(f"set up a reminder for '{reminder}'")
        
        for ledger in ledgers:
            if ledger["direction"] == "owe":
                task_details.append(f"logged ${ledger['amount']} with {ledger['contact']}")
            else:
                task_details.append(f"recorded ${ledger['amount']} from {ledger['contact']}")
        
        if task_details:
            if len(task_details) == 1:
                response_parts.append(f"I've {task_details[0]}.")
            elif len(task_details) == 2:
                response_parts.append(f"I've {task_details[0]} and {task_details[1]}.")
            else:
                formatted_details = f"I've {', '.join(task_details[:-1])}, and {task_details[-1]}."
                response_parts.append(formatted_details)
        
        # Add brief organizational note
        response_parts.append("Everything is organized and ready for you to access.")
        
        # Habit suggestion if applicable
        habit_question = None
        if habit_suggestion and habit_suggestion.get("is_habit", False):
            suggest_freq = habit_suggestion.get("suggest", "regularly")
            habit_question = f"Since you do this often, would you like me to set up a {suggest_freq} reminder?"
            response_parts.append(habit_question)
        
        # Follow-up question
        if not habit_question:
            if notes and reminders:
                follow_up = "What else would you like me to organize or remind you about?"
            elif notes:
                follow_up = "Anything else you'd like me to note for you?"
            elif reminders:
                follow_up = "Any other reminders you need?"
            elif ledgers:
                follow_up = "Other transactions to record?"
            else:
                follow_up = "What else can I help you with?"
        else:
            follow_up = habit_question
        
        # Extract actions
        actions = self._extract_actions_from_storage_results(storage_results)
        
        full_response = " ".join(response_parts)
        
        return {
            "response": full_response,
            "actions": actions,
            "follow_up": follow_up
        }
    
    def _generate_no_tasks_response(self) -> Dict[str, Any]:
        """Generate response when no tasks were processed."""
        return {
            "response": "I didn't catch any specific tasks from what you said. Could you please be more specific about what you'd like me to help you with? For example, 'remind me to...', 'note that...', or 'record that someone owes me...'",
            "actions": [],
            "follow_up": "What specific task can I help you with?"
        }

# Global instance
bloom_generator = BloomResponseGenerator() 