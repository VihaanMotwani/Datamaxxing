import os
import json
import argparse
from misinformation_detector import MisinformationDetector
from web_context_agent import WebContextAgent

class IntegratedMisinformationSystem:
    def __init__(self, groq_api_key=None, serpapi_key=None):
        """
        Initialize the integrated misinformation detection and context system.
        
        Args:
            groq_api_key (str, optional): Groq API key for LLM access
            serpapi_key (str, optional): SerpAPI key for web search
        """
        # Initialize API keys
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY")
        
        # Validate API keys
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
        
    def analyze_text(self, text_data, model=None, include_web_context=True):
        """
        Analyze text for misinformation and optionally get web context.
        
        Args:
            text_data (str): Text extracted from video
            model (str, optional): LLM model to use
            include_web_context (bool): Whether to include web context
            
        Returns:
            dict: Analysis results with misinformation detection and context
        """
        # Step 1: Detect potential misinformation
        print("Step 1: Analyzing text for potential misinformation...")
        detection_result = self.detector.analyze_text(text_data, model)
        
        # Initialize result with detection data
        result = {
            "misinformation_analysis": detection_result,
            "web_context": None,
            "timestamp": detection_result.get("timestamp", None)
        }
        
        # Step 2: Get web context if misinformation detected and context agent available
        if (include_web_context and self.context_agent and 
                detection_result.get("contains_misinformation", False)):
            
            print("Step 2: Getting additional context from the web...")
            
            # Extract the main claim(s) from the text
            claim = self._extract_claim(text_data, detection_result)
            
            # Fetch web context for the claim
            context_data = self.context_agent.analyze_claim(claim, format_output=False)
            result["web_context"] = context_data
        
        return result
    
    def _extract_claim(self, text_data, detection_result):
        """
        Extract the main claim from text data and detection results.
        
        Args:
            text_data (str): The original text data
            detection_result (dict): Misinformation detection results
            
        Returns:
            str: Extracted main claim for web search
        """
        # If the text is short, use it directly
        if len(text_data.split()) < 50:
            return text_data
        
        # Create a system prompt for claim extraction
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
        
        # Create user prompt with text and detection info
        user_prompt = f"""
        The following text was analyzed and potential misinformation was detected:
        
        TEXT: {text_data}
        
        DETECTION RESULTS:
        Detected criteria: {detection_result.get('detected_criteria', [])}
        Explanation: {detection_result.get('explanation', 'No explanation provided')}
        
        Extract the main claim that needs to be fact-checked. Return ONLY the claim without any additional text.
        """
        
        try:
            # Get claim extraction from LLM
            client = self.detector.client
            response = client.chat.completions.create(
                model=self.detector.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Extract the response
            claim = response.choices[0].message.content.strip()
            
            # If the extracted claim is too long, truncate it
            if len(claim.split()) > 50:
                claim = " ".join(claim.split()[:50]) + "..."
                
            return claim
            
        except Exception as e:
            # If extraction fails, use a substring of the original text
            words = text_data.split()
            if len(words) > 50:
                return " ".join(words[:50]) + "..."
            return text_data
    
    def format_results(self, results):
        """
        Format analysis results for display.
        
        Args:
            results (dict): Combined analysis results
            
        Returns:
            str: Formatted text for display
        """
        output = []
        
        # Add header
        output.append("=" * 80)
        output.append("MISINFORMATION ANALYSIS RESULTS")
        output.append("=" * 80)
        
        # Add misinformation analysis
        detection = results.get("misinformation_analysis", {})
        
        if detection.get("contains_misinformation") is True:
            output.append("\n⚠️  POTENTIAL MISINFORMATION DETECTED")
            output.append(f"Confidence: {detection.get('confidence_score', 0)*100:.1f}%")
        elif detection.get("contains_misinformation") is False:
            output.append("\n✅ NO MISINFORMATION DETECTED")
            output.append(f"Confidence: {detection.get('confidence_score', 0)*100:.1f}%")
        else:
            output.append("\n❓ ANALYSIS INCONCLUSIVE")
        
        # Add detected criteria
        if detection.get("detected_criteria"):
            output.append("\nDetected criteria:")
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
            for criterion in detection.get("detected_criteria", []):
                output.append(f"- {criterion}: {criteria_map.get(criterion, '')}")
        
        # Add explanation
        output.append("\nExplanation:")
        output.append(detection.get("explanation", "No explanation provided"))
        
        # Add web context if available
        web_context = results.get("web_context")
        if web_context:
            # Use the web context agent's formatter
            if self.context_agent:
                context_text = self.context_agent.format_context_for_display(web_context)
                output.append("\n" + context_text)
            else:
                # Simple formatting if web context agent not available
                output.append("\nADDITIONAL WEB CONTEXT:")
                
                if "error" in web_context:
                    output.append(f"Error retrieving context: {web_context['error']}")
                else:
                    output.append(f"\nClaim: {web_context.get('claim', 'Unknown')}")
                    output.append(f"\nSummary: {web_context.get('context_summary', 'No summary available')}")
                    
                    # Add sources
                    output.append("\nSources:")
                    for source in web_context.get("sources", []):
                        output.append(f"- {source.get('title', 'Unknown')}: {source.get('link', 'No link')}")
        
        return "\n".join(output)
    
    def interactive_analysis(self, text_data, model=None, include_web_context=True):
        """
        Run the analysis interactively, with prompts for additional context.
        
        Args:
            text_data (str): Text to analyze
            model (str, optional): LLM model to use
            include_web_context (bool): Whether to include web context
            
        Returns:
            dict: Analysis results
        """
        # Initial analysis
        results = self.analyze_text(text_data, model, include_web_context)
        
        # Format and print results
        formatted_results = self.format_results(results)
        print(formatted_results)
        
        # Check if user wants to provide additional context
        detection = results.get("misinformation_analysis", {})
        if detection.get("contains_misinformation") and detection.get("prompt_for_context"):
            print("\nDo you want to provide additional context? (yes/no)")
            user_response = input("> ").strip().lower()
            
            if user_response in ['yes', 'y']:
                print("\nPlease provide additional context:")
                additional_context = input("> ")
                
                # Combine original text with additional context
                combined_text = f"{text_data}\n\nADDITIONAL CONTEXT:\n{additional_context}"
                
                # Re-analyze with additional context
                print("\nRe-analyzing with additional context...")
                updated_results = self.analyze_text(combined_text, model, include_web_context)
                
                # Format and print updated results
                updated_formatted = self.format_results(updated_results)
                print(updated_formatted)
                
                return updated_results
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Integrated Misinformation Detection System')
    parser.add_argument('--text', help='Text to analyze')
    parser.add_argument('--file', help='File containing text to analyze')
    parser.add_argument('--groq-api-key', help='Groq API key')
    parser.add_argument('--serpapi-key', help='SerpAPI key for web search')
    parser.add_argument('--model', default='llama3-70b-8192', help='LLM model to use')
    parser.add_argument('--no-web-context', action='store_true', help='Skip retrieving web context')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    try:
        # Get text from file or command line
        text_data = args.text
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_data = f.read()
        
        if not text_data:
            # If no text is provided, prompt the user
            print("Enter text to analyze (press Ctrl+D or Ctrl+Z on an empty line to finish):")
            text_data = input("Enter: ")
        
        # Initialize the system
        system = IntegratedMisinformationSystem(
            groq_api_key=args.groq_api_key,
            serpapi_key=args.serpapi_key
        )
        
        # Run interactive analysis
        results = system.interactive_analysis(
            text_data, 
            model=args.model,
            include_web_context=not args.no_web_context
        )
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()