import os
import json
import argparse
import re
from groq import Groq

class MisinformationDetector:
    def __init__(self, api_key=None):
        """Initialize the misinformation detector with Groq API key."""
        # Use provided API key or get from environment variable
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided and not found in environment variables")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Default model to use
        self.model = "llama3-70b-8192"  # You can also use "llama3-8b-8192" for a smaller model
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text using multiple strategies.
        
        Args:
            text (str): Text that might contain JSON
            
        Returns:
            dict: Extracted JSON as dict, or None if extraction fails
        """
        # Strategy 1: Look for JSON between code blocks with language tag
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
        
        # Strategy 2: Look for JSON between any code blocks
        general_code_block = re.findall(r"```\s*(.+?)```", text, re.DOTALL)
        if general_code_block:
            try:
                return json.loads(general_code_block[0].strip())
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Look for JSON-like structure with braces
        try:
            # Find text between outermost braces
            brace_content = re.search(r"\{.+\}", text, re.DOTALL)
            if brace_content:
                json_candidate = brace_content.group(0)
                return json.loads(json_candidate)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Strategy 4: Try to parse the entire text as JSON
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
    
    def analyze_text(self, text_data, model=None):
        """
        Analyze text data for potential misinformation using Llama model through Groq API.
        
        Args:
            text_data (str): Text extracted from video (speech to text, OCR, captions)
            model (str, optional): Llama model to use. Defaults to llama3-70b-8192.
            
        Returns:
            dict: Analysis result containing misinformation assessment and explanation
        """
        # Use provided model or default
        model_to_use = model or self.model
        
        # Create the system prompt with misinformation detection criteria
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
        
        # Create the user prompt with the text data
        user_prompt = f"""
        Please analyze the following text extracted from a video for potential misinformation:

        TEXT DATA:
        {text_data}
        
        Return ONLY your assessment in JSON format as specified earlier with no additional text.
        """
        
        try:
            # Make API call to Groq - first without response_format (which may not be supported by all models)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Low temperature for more consistent analysis
                max_tokens=2000
                # No response_format parameter - some models may not support it
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response using our helper function
            analysis_result = self._extract_json_from_text(response_text)
            
            # If extraction failed, return error
            if not analysis_result:
                return self._create_default_analysis(
                    "Failed to extract valid JSON from response", 
                    response_text
                )
                
            # Ensure all expected fields are present with correct types
            required_fields = {
                'contains_misinformation': bool,
                'confidence_score': float,
                'detected_criteria': list,
                'explanation': str,
                'prompt_for_context': bool
            }
            
            # Fix or fill missing fields
            for field, field_type in required_fields.items():
                # If field is missing, set to default
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
                        # If conversion fails, set to default
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
            
            return analysis_result
            
        except Exception as e:
            # If an exception occurs during the API call or processing
            return self._create_default_analysis(
                f"Error during analysis: {str(e)}", 
                getattr(response, 'choices', [{}])[0].get('message', {}).get('content', 'No response text')
                if 'response' in locals() else "No response received"
            )
        
    def interactive_analysis(self, text_data, model=None):
        """
        Perform interactive analysis with user prompts for more context if needed.
        
        Args:
            text_data (str): Text extracted from video
            model (str, optional): Llama model to use
            
        Returns:
            dict: Final analysis result
        """
        # Initial analysis
        result = self.analyze_text(text_data, model)
        
        # Print the analysis
        print("\n=== MISINFORMATION ANALYSIS RESULTS ===")
        print(f"Contains misinformation: {result.get('contains_misinformation', 'Unknown')}")
        if result.get('confidence_score') is not None:
            print(f"Confidence score: {result['confidence_score']:.2f}")
        
        if result.get('detected_criteria'):
            print("\nDetected criteria:")
            criteria_map = {
                1: "Factual Accuracy",
                2: "Source Credibility",
                3: "Logical Consistency",
                4: "Scientific Consensus",
                5: "Context Manipulation",
                6: "Statistical Misrepresentation",
                7: "Emotional Manipulation",
                8: "Unverifiable Claims",
                9: "Conspiracy Narratives",
                10: "False Equivalence"
            }
            for criterion in result['detected_criteria']:
                print(f"- {criterion}: {criteria_map.get(criterion, '')}")
        
        print("\nExplanation:")
        print(result.get('explanation', 'No explanation provided'))
        
        # Print error if present
        if 'error' in result and result['error']:
            print("\nError encountered:")
            print(result['error'])
            
            # If there was an error, print the raw response for debugging
            if 'raw_response' in result and result['raw_response']:
                print("\nRaw response from model:")
                print(result['raw_response'][:500] + "..." if len(result['raw_response']) > 500 else result['raw_response'])
        
        # Prompt for more context if needed
        if result.get('contains_misinformation') and result.get('prompt_for_context'):
            print("\nDo you want more context? (yes/no)")
            user_response = input("> ").strip().lower()
            
            if user_response in ['yes', 'y']:
                print("\nPlease provide additional context:")
                additional_context = input("> ")
                
                # Combine original text with additional context
                combined_text = f"{text_data}\n\nADDITIONAL CONTEXT:\n{additional_context}"
                
                # Re-analyze with additional context
                updated_result = self.analyze_text(combined_text, model)
                
                # Print updated analysis
                print("\n=== UPDATED ANALYSIS WITH ADDITIONAL CONTEXT ===")
                print(f"Contains misinformation: {updated_result.get('contains_misinformation', 'Unknown')}")
                if updated_result.get('confidence_score') is not None:
                    print(f"Confidence score: {updated_result['confidence_score']:.2f}")
                
                print("\nUpdated explanation:")
                print(updated_result.get('explanation', 'No explanation provided'))
                
                return updated_result
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Detect misinformation in video text data using Llama models via Groq API')
    parser.add_argument('--api-key', help='Groq API key (defaults to GROQ_API_KEY env variable)')
    parser.add_argument('--model', default='llama3-70b-8192', help='Llama model to use (default: llama3-70b-8192)')
    parser.add_argument('--file', help='Path to text file containing video text data')
    parser.add_argument('--text', help='Direct text input for analysis')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # Get text data from file or direct input
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text_data = f.read()
    elif args.text:
        text_data = args.text
    else:
        # If no input is provided, prompt the user
        print("Enter or paste the text data extracted from video (press Ctrl+D or Ctrl+Z on an empty line to finish):")
        text_data = input()
    
    # Initialize detector and run analysis
    detector = MisinformationDetector(api_key=args.api_key)
    
    if args.debug:
        print(f"Using model: {args.model}")
        print(f"API key is set: {'Yes' if detector.api_key else 'No'}")
        print(f"Text data length: {len(text_data)} characters")
        
    detector.interactive_analysis(text_data, model=args.model)

if __name__ == "__main__":
    main()