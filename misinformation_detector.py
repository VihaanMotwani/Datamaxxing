import os
import json
import re
from groq import Groq

class MisinformationDetector:
    def __init__(self, api_key=None):
        """Initialize the misinformation detector with Groq API key."""

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided and not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        
        self.model = "llama3-70b-8192"  # ALternatively, "llama3-8b-8192" (smaller model)
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text using multiple strategies.
        
        Args:
            text (str): Text that might contain JSON
            
        Returns:
            dict: Extracted JSON as dict, or None if extraction fails
        """
        # Look for JSON between code blocks with language tag
        code_block_patterns = [
            r"```json\s*(.+?)```",  # ```json {...} ```
            r"```javascript\s*(.+?)```",  # ```javascript {...} ```
            r"```js\s*(.+?)```",  # ```js {...} ```
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0].strip())
                except json.JSONDecodeError:
                    continue
        
        # Look for JSON between any code blocks
        general_code_block = re.findall(r"```\s*(.+?)```", text, re.DOTALL)
        if general_code_block:
            try:
                return json.loads(general_code_block[0].strip())
            except json.JSONDecodeError:
                pass
        
        # Look for JSON-like structure with braces
        try:
            # Find text between outermost braces
            brace_content = re.search(r"\{.+\}", text, re.DOTALL)
            if brace_content:
                json_candidate = brace_content.group(0)
                return json.loads(json_candidate)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try to parse the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # If all strategies fail, return None
        return None
    
    def _create_default_analysis(self, error_message="Unknown error", raw_response=""):
        """Create a default analysis result with error information."""
        return {
            "error": error_message,
            "raw_response": raw_response,
            "contains_misinformation": None,
            "confidence_score": None,
            "detected_criteria": [],
            "explanation": f"Error in analysis: {error_message}. Please verify manually.",
            "prompt_for_context": True
        }
    
    def _validate_and_fix_fields(self, analysis_result):
        """Validate and fix fields in the analysis result."""
        required_fields = {
            'contains_misinformation': bool,
            'confidence_score': float,
            'detected_criteria': list,
            'explanation': str,
            'prompt_for_context': bool
        }
        
        # Fix or fill missing fields
        for field, field_type in required_fields.items():
            if field not in analysis_result:
                if field_type == bool:
                    analysis_result[field] = False
                elif field_type == float:
                    analysis_result[field] = 0.0
                elif field_type == list:
                    analysis_result[field] = []
                elif field_type == str:
                    analysis_result[field] = "No explanation provided"
                else:
                    analysis_result[field] = None
            # If field is present but wrong type, try to convert
            elif analysis_result[field] is not None and not isinstance(analysis_result[field], field_type):
                try:
                    if field_type == bool:
                        # Handle string representations of booleans
                        if isinstance(analysis_result[field], str):
                            analysis_result[field] = analysis_result[field].lower() in ['true', 'yes', '1']
                        else:
                            analysis_result[field] = bool(analysis_result[field])
                    elif field_type == float:
                        analysis_result[field] = float(analysis_result[field])
                    elif field_type == list and isinstance(analysis_result[field], str):
                        # Try to convert string representations of lists
                        if ',' in analysis_result[field]:
                            analysis_result[field] = [int(x.strip()) for x in analysis_result[field].split(',') if x.strip().isdigit()]
                        else:
                            analysis_result[field] = []
                    elif field_type == str:
                        analysis_result[field] = str(analysis_result[field])
                except (ValueError, TypeError):
                    if field_type == bool:
                        analysis_result[field] = False
                    elif field_type == float:
                        analysis_result[field] = 0.0
                    elif field_type == list:
                        analysis_result[field] = []
                    elif field_type == str:
                        analysis_result[field] = "No explanation provided"
        
        # Convert any string representations of lists to actual lists if needed
        if isinstance(analysis_result['detected_criteria'], str):
            try:
                # Try to interpret as a comma-separated list
                analysis_result['detected_criteria'] = [
                    int(x.strip()) for x in analysis_result['detected_criteria'].split(',')
                    if x.strip().isdigit() and 1 <= int(x.strip()) <= 10
                ]
            except:
                analysis_result['detected_criteria'] = []
        
        # Filter out any criteria that are not valid numbers between 1-10
        analysis_result['detected_criteria'] = [
            c for c in analysis_result['detected_criteria'] 
            if isinstance(c, int) and 1 <= c <= 10
        ]
        
        # Ensure confidence score is between 0 and 1
        if 'confidence_score' in analysis_result:
            try:
                score = float(analysis_result['confidence_score'])
                analysis_result['confidence_score'] = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                analysis_result['confidence_score'] = 0.0
    
    def analyze_text(self, text_data, model=None):
        """
        Analyze text data for potential misinformation using Llama model through Groq API.
        
        Args:
            text_data (str): Text extracted from video (speech to text, OCR, captions)
            model (str, optional): Llama model to use. Defaults to llama3-70b-8192.
            
        Returns:
            dict: Analysis result containing misinformation assessment and explanation
        """
        model_to_use = model or self.model
        
        system_prompt = """
        You are an expert at detecting misinformation in content. Your task is to analyze the text provided
        and determine if it might contain misinformation. Use the following criteria for your assessment:

        MISINFORMATION DETECTION CRITERIA:
        1. Factual Accuracy: Does the content include demonstrably false claims or inaccurate information?
        2. Source Credibility: Are claims attributed to unreliable, non-existent, or misrepresented sources?
        3. Logical Consistency: Are there logical fallacies, contradictions, or inconsistencies in the arguments?
        4. Scientific Consensus: Do claims contradict well-established scientific consensus without substantial evidence?
        5. Context Manipulation: Is information presented out of context or in a misleading way?
        6. Statistical Misrepresentation: Are statistics or data misrepresented, cherry-picked, or manipulated?
        7. Emotional Manipulation: Does the content rely heavily on emotional appeals rather than factual evidence?
        8. Unverifiable Claims: Are extraordinary claims made without corresponding evidence?
        9. Conspiracy Narratives: Does the content promote unfounded conspiracy theories?
        10. False Equivalence: Does the content present unequal positions as if they have equal merit?

        Your analysis MUST be provided in valid JSON format with the following fields:
        {
          "contains_misinformation": boolean (true if potential misinformation detected, false otherwise),
          "confidence_score": number from 0 to 1 indicating your confidence level,
          "detected_criteria": list of criteria numbers (1-10) that were detected,
          "explanation": detailed explanation of your assessment,
          "prompt_for_context": boolean indicating if more context would help clarify (true if needed)
        }
        
        Provide ONLY the JSON object in your response with no additional text or explanations.
        """
        
        user_prompt = f"""
        Please analyze the following text extracted from a video for potential misinformation:

        TEXT DATA:
        {text_data}
        
        Return ONLY your assessment in JSON format as specified earlier with no additional text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            analysis_result = self._extract_json_from_text(response_text)
            
            if not analysis_result:
                return self._create_default_analysis(
                    "Failed to extract valid JSON from response", 
                    response_text
                )
                
            self._validate_and_fix_fields(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            return self._create_default_analysis(
                f"Error during analysis: {str(e)}", 
                getattr(response, 'choices', [{}])[0].get('message', {}).get('content', 'No response text')
                if 'response' in locals() else "No response received"
            )
    
    def process_json_input(self, json_data):
        """
        Process input from JSON data containing video text and transcription.
        
        Args:
            json_data (dict): JSON data with 'transcription' and 'extracted_text' fields
            
        Returns:
            str: Combined text for analysis
        """
        transcription = json_data.get("transcription", "")
        extracted_text = json_data.get("extracted_text", "")
        
        combined_text = ""
        
        if transcription:
            combined_text += "TRANSCRIPTION:\n" + transcription + "\n\n"
            
        if extracted_text:
            combined_text += "EXTRACTED TEXT (OCR):\n" + extracted_text
            
        return combined_text.strip()
    
    def extract_claim(self, text_data, detection_result):
        """
        Extract the main claim from text data and detection results.
        
        Args:
            text_data (str): The original text data
            detection_result (dict): Misinformation detection results
            
        Returns:
            str: Extracted main claim for web search
        """
        if len(text_data.split()) < 50:
            return text_data
        
        system_prompt = """
        You are an expert at identifying the main claims in text. 
        Extract the central claim that was identified as potential misinformation.
        
        Focus on specific factual assertions that can be verified, especially those that:
        1. Make definitive statements about facts
        2. Cite statistics, research or studies
        3. Make health, medical, or scientific claims
        4. Reference historical events
        5. Make causal claims (X causes Y)
        
        Return ONLY the extracted claim as a concise statement without any additional commentary.
        If there are multiple claims, focus on the most significant one that appears to be potentially false.
        """
        
        user_prompt = f"""
        The following text was analyzed and potential misinformation was detected:
        
        TEXT: {text_data}
        
        DETECTION RESULTS:
        Detected criteria: {detection_result.get('detected_criteria', [])}
        Explanation: {detection_result.get('explanation', 'No explanation provided')}
        
        Extract the main claim that needs to be fact-checked. Return ONLY the claim without any additional text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            claim = response.choices[0].message.content.strip()
            
            if len(claim.split()) > 50:
                claim = " ".join(claim.split()[:50]) + "..."
                
            return claim
            
        except Exception as e:
            words = text_data.split()
            if len(words) > 50:
                return " ".join(words[:50]) + "..."
            return text_data