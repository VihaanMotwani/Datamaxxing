import os
import json
import requests
import argparse
from groq import Groq
from urllib.parse import quote_plus
from datetime import datetime

class WebContextAgent:
    def __init__(self, api_key=None, search_api_key=None):
        """
        Initialize the Web Context Agent with necessary API keys.
        
        Args:
            api_key (str, optional): Groq API key for LLM access
            search_api_key (str, optional): SerpAPI key for web search
        """
        # Initialize Groq API for LLM access
        self.groq_api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key not provided and not found in environment variables")
        
        # Initialize search API
        self.search_api_key = search_api_key or os.environ.get("SERPAPI_KEY")
        if not self.search_api_key:
            raise ValueError("Search API key not provided and not found in environment variables")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.groq_api_key)
        
        # Default LLM model
        self.model = "llama3-70b-8192"
        
        # List of known reliable source domains for preferential ranking
        self.reliable_sources = [
            # Major news organizations
            "reuters.com", "apnews.com", "bloomberg.com", "bbc.com", "bbc.co.uk",
            "npr.org", "pbs.org", "economist.com", "wsj.com", "nytimes.com",
            "washingtonpost.com", "ft.com", "cnn.com", "nbcnews.com", "cbsnews.com",
            "abcnews.go.com", "time.com", "theatlantic.com", "newyorker.com",
            
            # Science/Medical sources
            "nature.com", "science.org", "sciencedaily.com", "nih.gov", "who.int",
            "mayoclinic.org", "hopkinsmedicine.org", "harvard.edu", "stanford.edu",
            "mit.edu", "cdc.gov", "healthline.com", "medlineplus.gov", "sciencemag.org",
            
            # Fact-checking sites
            "snopes.com", "factcheck.org", "politifact.com", "aap.com.au", "fullfact.org",
            "checkyourfact.com", "truthorfiction.com", "factcheckni.org",
            
            # Education/Reference
            "britannica.com", "edu.gov", "smithsonianmag.com", "nationalgeographic.com",
            "history.com", "jstor.org", "scientificamerican.com", "newscientist.com",
            
            # Government sources
            "gov", "europa.eu", "un.org", "nasa.gov", "noaa.gov", "epa.gov", "fda.gov",
            
            # Research institutions
            "rand.org", "brookings.edu", "pewresearch.org", "worldbank.org", "imf.org"
        ]
    
    def search_web(self, query, num_results=5):
        """
        Search the web for information related to a query.
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return
            
        Returns:
            list: Search results with title, snippet, source, and link
        """
        # Prepare search query
        encoded_query = quote_plus(query)
        search_url = f"https://serpapi.com/search.json?engine=google&q={encoded_query}&api_key={self.search_api_key}&num={num_results*2}"  # Request more to filter
        
        try:
            # Make request to search API
            response = requests.get(search_url)
            data = response.json()
            
            if "error" in data:
                return {"error": data["error"]}
            
            # Extract organic search results
            results = []
            if "organic_results" in data:
                for result in data["organic_results"]:
                    # Extract domain from link
                    domain = result.get("domain", "")
                    
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "source": domain,
                        "link": result.get("link", ""),
                        "is_reliable": any(source in domain for source in self.reliable_sources),
                        "position": result.get("position", 999)
                    })
            
            # Sort results to prioritize reliable sources
            results.sort(key=lambda x: (not x["is_reliable"], x["position"]))
            
            # Return the requested number of results
            return results[:num_results]
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def evaluate_sources(self, search_results):
        """
        Evaluate the reliability and relevance of search results.
        
        Args:
            search_results (list): List of search results to evaluate
            
        Returns:
            list: Same results with reliability ratings added
        """
        if not search_results or "error" in search_results:
            return search_results
        
        # Prepare input for LLM evaluation
        sources_text = ""
        for i, result in enumerate(search_results):
            sources_text += f"Source {i+1}: {result['title']}\n"
            sources_text += f"URL: {result['link']}\n"
            sources_text += f"Source domain: {result['source']}\n"
            sources_text += f"Snippet: {result['snippet']}\n\n"
        
        # Define the system prompt for source evaluation
        system_prompt = """
        You are an expert at evaluating information sources for reliability and credibility.
        Assess each source based on these criteria:
        
        1. Authority: Is it from a reputable organization or expert in the relevant field?
        2. Accuracy: Does it contain factual information supported by evidence?
        3. Objectivity: Does it present information in a balanced way or show bias?
        4. Currency: Is it up-to-date and relevant?
        5. Coverage: Does it provide comprehensive information on the topic?
        
        For each source, provide:
        1. Reliability Score (1-10)
        2. Brief reasoning for the score
        3. Any potential biases or limitations
        
        Return your analysis in JSON format:
        {
          "evaluations": [
            {
              "source_num": 1,
              "reliability_score": 8,
              "reasoning": "Brief explanation here",
              "potential_bias": "Any potential bias or limitation"
            },
            ...
          ]
        }
        """
        
        user_prompt = f"""
        Please evaluate the following sources for reliability and credibility:
        
        {sources_text}
        
        Provide only the JSON response with your evaluations.
        """
        
        try:
            # Get LLM evaluation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            
            # Extract and parse response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Try to find JSON in code blocks first
                if "```json" in response_text and "```" in response_text.split("```json")[1]:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    evaluations = json.loads(json_str)
                elif "```" in response_text:
                    # Look for JSON in any code block
                    json_str = response_text.split("```")[1].strip()
                    evaluations = json.loads(json_str)
                else:
                    # Try to parse the whole response
                    evaluations = json.loads(response_text)
                
                # Add evaluations to search results
                if "evaluations" in evaluations:
                    for i, eval_data in enumerate(evaluations["evaluations"]):
                        if i < len(search_results):
                            search_results[i]["reliability_score"] = eval_data.get("reliability_score", 5)
                            search_results[i]["evaluation_reasoning"] = eval_data.get("reasoning", "")
                            search_results[i]["potential_bias"] = eval_data.get("potential_bias", "")
            
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                # If parsing fails, add default evaluations
                for result in search_results:
                    result["reliability_score"] = 5  # Default middle score
                    result["evaluation_reasoning"] = "Automatic evaluation based on source domain"
                    result["potential_bias"] = "Unknown - evaluation failed"
            
            return search_results
            
        except Exception as e:
            # If evaluation fails, return original results
            return search_results
    
    def fetch_context(self, claim, search_results=None):
        """
        Fetch and synthesize context from multiple sources for a claim.
        
        Args:
            claim (str): The claim to investigate
            search_results (list, optional): Pre-fetched search results
            
        Returns:
            dict: Synthesized context with sources and balanced perspective
        """
        # If search results not provided, search the web
        if not search_results:
            search_results = self.search_web(claim)
        
        # If search failed, return error
        if isinstance(search_results, dict) and "error" in search_results:
            return {
                "error": search_results["error"],
                "context": "Could not retrieve context due to search error."
            }
        
        # Evaluate sources if not already evaluated
        if search_results and "reliability_score" not in search_results[0]:
            search_results = self.evaluate_sources(search_results)
        
        # Prepare source information for the LLM
        sources_info = ""
        for i, result in enumerate(search_results):
            sources_info += f"Source {i+1}: {result['title']}\n"
            sources_info += f"URL: {result['link']}\n"
            sources_info += f"Reliability score: {result.get('reliability_score', 'Not evaluated')}/10\n"
            sources_info += f"Snippet: {result['snippet']}\n\n"
        
        # Create system prompt for balanced context synthesis
        system_prompt = """
        You are an impartial research assistant tasked with providing balanced context around potentially controversial claims.
        Your goal is to synthesize information from multiple sources to help people understand different perspectives.
        
        IMPORTANT GUIDELINES:
        1. Present information from multiple perspectives, giving fair representation to different viewpoints
        2. Prioritize information from more reliable sources, but include diverse viewpoints
        3. Clearly distinguish between well-established facts and disputed claims
        4. Do not make definitive truth judgments; present evidence and let readers decide
        5. Avoid using loaded language or expressing your own opinion
        6. Include relevant scientific consensus where applicable
        7. Note limitations in available information
        8. Include source citations for all key points
        
        For claims involving science or health:
        - Emphasize peer-reviewed research and scientific consensus
        - Note sample sizes and study limitations
        - Distinguish between correlation and causation
        
        For claims involving politics or social issues:
        - Present perspectives from across the political spectrum
        - Avoid partisan framing
        - Include relevant historical context
        
        Format your response as JSON with these fields:
        {
          "claim": "The original claim",
          "context_summary": "Brief neutral summary of what's known about the claim",
          "perspectives": [
            {
              "viewpoint": "Description of perspective 1",
              "supporting_evidence": "Evidence for this perspective",
              "limitations": "Limitations of this evidence",
              "sources": [1, 3] // Source numbers that support this perspective
            },
            {
              "viewpoint": "Description of perspective 2",
              "supporting_evidence": "Evidence for this perspective",
              "limitations": "Limitations of this evidence",
              "sources": [2, 4] // Source numbers that support this perspective
            }
          ],
          "scientific_consensus": "Description of any relevant scientific consensus (if applicable)",
          "conclusion": "Balanced summary without making definitive truth claims",
          "information_gaps": "Important missing information needed for full context"
        }
        """
        
        # Create user prompt
        user_prompt = f"""
        Please provide balanced context for this claim:
        
        CLAIM: {claim}
        
        SOURCES:
        {sources_info}
        
        Synthesize information from these sources to provide balanced context. 
        Return only the JSON response with the context information.
        """
        
        try:
            # Get context synthesis from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            # Extract and parse response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Try to find JSON in code blocks first
                if "```json" in response_text and "```" in response_text.split("```json")[1]:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    context_data = json.loads(json_str)
                elif "```" in response_text:
                    # Look for JSON in any code block
                    json_str = response_text.split("```")[1].strip()
                    context_data = json.loads(json_str)
                else:
                    # Try to parse the whole response
                    context_data = json.loads(response_text)
                
                # Add source information to the context data
                context_data["sources"] = [
                    {
                        "number": i+1,
                        "title": result["title"],
                        "link": result["link"],
                        "reliability_score": result.get("reliability_score", "Not evaluated")
                    } for i, result in enumerate(search_results)
                ]
                
                # Add timestamp
                context_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return context_data
                
            except (json.JSONDecodeError, KeyError) as e:
                # If parsing fails, create a simple context response
                return {
                    "error": f"Failed to parse context synthesis: {str(e)}",
                    "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    "sources": [
                        {
                            "number": i+1,
                            "title": result["title"],
                            "link": result["link"]
                        } for i, result in enumerate(search_results)
                    ]
                }
                
        except Exception as e:
            return {
                "error": f"Context synthesis failed: {str(e)}",
                "sources": [
                    {
                        "number": i+1,
                        "title": result["title"],
                        "link": result["link"]
                    } for i, result in enumerate(search_results)
                ] if search_results else []
            }
    
    def format_context_for_display(self, context_data):
        """
        Format the context data for human-readable display.
        
        Args:
            context_data (dict): The context data from fetch_context
            
        Returns:
            str: Formatted context for display
        """
        if "error" in context_data:
            return f"Error retrieving context: {context_data['error']}\n\n"
        
        output = []
        
        # Add header
        output.append("=" * 80)
        output.append("ADDITIONAL CONTEXT FROM WEB SOURCES")
        output.append("=" * 80)
        
        # Add claim
        output.append(f"\nCLAIM: {context_data.get('claim', 'No claim provided')}\n")
        
        # Add context summary
        output.append("SUMMARY:")
        output.append(context_data.get('context_summary', 'No summary available'))
        output.append("")
        
        # Add perspectives
        if "perspectives" in context_data and context_data["perspectives"]:
            output.append("DIFFERENT PERSPECTIVES:")
            for i, perspective in enumerate(context_data["perspectives"]):
                output.append(f"\nPerspective {i+1}: {perspective.get('viewpoint', '')}")
                output.append(f"Supporting evidence: {perspective.get('supporting_evidence', '')}")
                output.append(f"Limitations: {perspective.get('limitations', '')}")
                
                # Add source references
                if "sources" in perspective and perspective["sources"]:
                    source_refs = []
                    for src_num in perspective["sources"]:
                        # Find the corresponding source in the sources list
                        for source in context_data.get("sources", []):
                            if source.get("number") == src_num:
                                source_refs.append(f"Source {src_num} ({source.get('title', 'Unknown')})")
                                break
                    if source_refs:
                        output.append(f"Sources: {', '.join(source_refs)}")
                output.append("")
        
        # Add scientific consensus if available
        if "scientific_consensus" in context_data and context_data["scientific_consensus"]:
            output.append("SCIENTIFIC CONSENSUS:")
            output.append(context_data["scientific_consensus"])
            output.append("")
        
        # Add conclusion
        if "conclusion" in context_data and context_data["conclusion"]:
            output.append("CONCLUSION:")
            output.append(context_data["conclusion"])
            output.append("")
        
        # Add information gaps
        if "information_gaps" in context_data and context_data["information_gaps"]:
            output.append("INFORMATION GAPS:")
            output.append(context_data["information_gaps"])
            output.append("")
        
        # Add sources list
        output.append("SOURCES:")
        for source in context_data.get("sources", []):
            reliability = f" (Reliability score: {source.get('reliability_score', 'Not evaluated')}/10)" if "reliability_score" in source else ""
            output.append(f"{source.get('number', '?')}. {source.get('title', 'Unknown')}{reliability}")
            output.append(f"   Link: {source.get('link', 'No link available')}")
        
        # Add timestamp
        if "timestamp" in context_data:
            output.append(f"\nContext retrieved on: {context_data['timestamp']}")
        
        return "\n".join(output)
    
    def analyze_claim(self, claim, format_output=True):
        """
        Analyze a claim by searching the web and providing balanced context.
        
        Args:
            claim (str): The claim to analyze
            format_output (bool): Whether to return formatted text (True) or raw data (False)
            
        Returns:
            str or dict: Formatted context text or raw context data
        """
        print(f"Analyzing claim: {claim}")
        print("Searching for relevant information...")
        
        # Search the web
        search_results = self.search_web(claim)
        
        if isinstance(search_results, dict) and "error" in search_results:
            print(f"Search error: {search_results['error']}")
            if format_output:
                return f"Error searching for context: {search_results['error']}"
            return {"error": search_results["error"]}
        
        print(f"Found {len(search_results)} relevant sources.")
        print("Evaluating source reliability...")
        
        # Evaluate sources
        evaluated_results = self.evaluate_sources(search_results)
        
        print("Synthesizing context from multiple sources...")
        
        # Fetch context
        context_data = self.fetch_context(claim, evaluated_results)
        
        if "error" in context_data:
            print(f"Context synthesis error: {context_data['error']}")
        else:
            print("Context analysis complete.")
        
        # Return formatted or raw data
        if format_output:
            return self.format_context_for_display(context_data)
        return context_data

def main():
    parser = argparse.ArgumentParser(description='Web Context Agent - Fetch balanced context for claims')
    parser.add_argument('--claim', help='The claim to analyze')
    parser.add_argument('--groq-api-key', help='Groq API key (defaults to GROQ_API_KEY env variable)')
    parser.add_argument('--search-api-key', help='SerpAPI key (defaults to SERPAPI_KEY env variable)')
    parser.add_argument('--raw', action='store_true', help='Output raw JSON data instead of formatted text')
    
    args = parser.parse_args()
    
    try:
        # Initialize the agent
        agent = WebContextAgent(api_key=args.groq_api_key, search_api_key=args.search_api_key)
        
        # Get claim from args or prompt
        claim = args.claim
        if not claim:
            claim = input("Enter the claim to analyze: ")
        
        # Analyze the claim
        result = agent.analyze_claim(claim, format_output=not args.raw)
        
        # Print result
        if args.raw:
            print(json.dumps(result, indent=2))
        else:
            print(result)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()