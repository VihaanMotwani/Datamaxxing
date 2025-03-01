import os
import json
from datetime import datetime
from misinformation_detector import MisinformationDetector
from web_context_agent import WebContextAgent

class IntegratedSystem:
    def __init__(self, groq_api_key=None, serpapi_key=None):
        """
        Initialize the integrated misinformation detection and context system.
        
        Args:
            groq_api_key (str, optional): Groq API key for LLM access
            serpapi_key (str, optional): SerpAPI key for web search
        """
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY")
        
        if not self.groq_api_key:
            raise ValueError("Groq API key not provided and not found in environment variables")
        
        # Initialize components
        self.detector = MisinformationDetector(api_key=self.groq_api_key)
        
        # Only initialize web context agent if search API key is available
        self.context_agent = None
        if self.serpapi_key:
            self.context_agent = WebContextAgent(
                api_key=self.groq_api_key, 
                search_api_key=self.serpapi_key
            )
    
    def analyze_json_file(self, json_file_path, model=None, include_web_context=True):
        """
        Analyze text data from a JSON file for potential misinformation.
        
        Args:
            json_file_path (str): Path to JSON file with video data
            model (str, optional): Llama model to use
            include_web_context (bool): Whether to include web context
            
        Returns:
            dict: Analysis result with misinformation assessment and web context
        """
        try:
            # Read the JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            # Process the JSON data to get combined text
            combined_text = self.detector.process_json_input(json_data)
            
            if not combined_text:
                return {
                    "error": "No text data found in JSON file",
                    "contains_misinformation": None,
                    "confidence_score": None,
                    "detected_criteria": [],
                    "explanation": "Could not analyze empty text data",
                    "web_context": None
                }
            
            # Step 1: Analyze text for potential misinformation
            print("Step 1: Analyzing text for potential misinformation...")
            detection_result = self.detector.analyze_text(combined_text, model)
            
            # Initialize result with detection data
            result = {
                "misinformation_analysis": detection_result,
                "web_context": None,
                "input_file": os.path.basename(json_file_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Print misinformation analysis results
            self.print_analysis_result(detection_result)
            
            # Step 2: Get web context if misinformation detected and context agent available
            if (include_web_context and self.context_agent and 
                    detection_result.get("contains_misinformation", False) is True):
                
                print("\nStep 2: Getting additional context from the web...")
                
                # Extract the main claim(s) from the text
                claim = self.detector.extract_claim(combined_text, detection_result)
                print(f"Extracted claim: {claim}")
                
                # Fetch web context for the claim
                context_data = self.context_agent.fetch_context(claim)
                result["web_context"] = context_data
                
                # Print web context
                if context_data and "error" not in context_data:
                    print(self.context_agent.format_context_for_display(context_data))
                elif "error" in context_data:
                    print(f"Error retrieving web context: {context_data['error']}")
                else:
                    print("No web context retrieved")
            elif detection_result.get("contains_misinformation", False) is True and not self.context_agent:
                print("\nStep 2 skipped: Web context agent not available (SERPAPI_KEY not set)")
            elif detection_result.get("contains_misinformation", False) is not True:
                print("\nStep 2 skipped: No misinformation detected")
            elif not include_web_context:
                print("\nStep 2 skipped: Web context disabled")
            
            return result
            
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_file_path}")
            return {
                "error": f"JSON file not found: {json_file_path}",
                "misinformation_analysis": {"contains_misinformation": None},
                "web_context": None
            }
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {json_file_path}")
            return {
                "error": f"Invalid JSON format in file: {json_file_path}",
                "misinformation_analysis": {"contains_misinformation": None},
                "web_context": None
            }
        except Exception as e:
            print(f"Error analyzing JSON file: {str(e)}")
            return {
                "error": f"Error analyzing JSON file: {str(e)}",
                "misinformation_analysis": {"contains_misinformation": None},
                "web_context": None
            }
    
    def analyze_text(self, text_data, model=None, include_web_context=True):
        """
        Analyze raw text for potential misinformation.
        
        Args:
            text_data (str): Text to analyze
            model (str, optional): Llama model to use
            include_web_context (bool): Whether to include web context
            
        Returns:
            dict: Analysis result with misinformation detection and context
        """
        # Step 1: Detect potential misinformation
        print("Step 1: Analyzing text for potential misinformation...")
        detection_result = self.detector.analyze_text(text_data, model)
        
        # Initialize result with detection data
        result = {
            "misinformation_analysis": detection_result,
            "web_context": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Print misinformation analysis results
        self.print_analysis_result(detection_result)
        
        # Step 2: Get web context if misinformation detected and context agent available
        if (include_web_context and self.context_agent and 
                detection_result.get("contains_misinformation", False) is True):
            
            print("\nStep 2: Getting additional context from the web...")
            
            # Extract the main claim(s) from the text
            claim = self.detector.extract_claim(text_data, detection_result)
            print(f"Extracted claim: {claim}")
            
            # Fetch web context for the claim
            context_data = self.context_agent.fetch_context(claim)
            result["web_context"] = context_data
            
            # Print web context
            if context_data and "error" not in context_data:
                print(self.context_agent.format_context_for_display(context_data))
            elif "error" in context_data:
                print(f"Error retrieving web context: {context_data['error']}")
            else:
                print("No web context retrieved")
        elif detection_result.get("contains_misinformation", False) is True and not self.context_agent:
            print("\nStep 2 skipped: Web context agent not available (SERPAPI_KEY not set)")
        elif detection_result.get("contains_misinformation", False) is not True:
            print("\nStep 2 skipped: No misinformation detected")
        elif not include_web_context:
            print("\nStep 2 skipped: Web context disabled")
        
        return result
    
    def print_analysis_result(self, result):
        """Print formatted analysis result."""
        print("\n" + "=" * 80)
        print("MISINFORMATION ANALYSIS RESULTS")
        print("=" * 80)
        
        if result.get('contains_misinformation') is True:
            print("\n⚠️  POTENTIAL MISINFORMATION DETECTED")
        elif result.get('contains_misinformation') is False:
            print("\n✅ NO MISINFORMATION DETECTED")
        else:
            print("\n❓ ANALYSIS INCONCLUSIVE")
            
        if result.get('confidence_score') is not None:
            print(f"Confidence: {result['confidence_score']*100:.1f}%")
        
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