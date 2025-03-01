#!/usr/bin/env python3
"""
TruthLens: A system for detecting misinformation in video content and providing web context.

This script serves as the main entry point for the TruthLens system.
"""

import os
import json
import argparse
from integrated_system import IntegratedSystem
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Main entry point for TruthLens."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='TruthLens: Detect misinformation in video text and provide context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single JSON file
  python main.py --json-file video_data.json
  
  # Process all JSON files in a directory
  python main.py --json-dir videos_folder
  
  # Analyze raw text
  python main.py --text "Scientists have found that lemon water cures diabetes."
  
  # Save analysis results to a directory
  python main.py --json-file video_data.json --output-dir results
"""
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--json-file', help='Path to a JSON file containing video text data')
    input_group.add_argument('--json-dir', help='Directory path to process multiple JSON files')
    input_group.add_argument('--text', help='Direct text input for analysis')
    input_group.add_argument('--text-file', help='Path to text file containing content to analyze')
    
    # API keys
    api_group = parser.add_argument_group('API Keys')
    api_group.add_argument('--groq-api-key', help='Groq API key for LLM access (or set GROQ_API_KEY env variable)')
    api_group.add_argument('--serpapi-key', help='SerpAPI key for web search (or set SERPAPI_KEY env variable)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--model', default='llama3-70b-8192', help='LLM model to use (default: llama3-70b-8192)')
    analysis_group.add_argument('--no-web-context', action='store_true', help='Skip retrieving web context')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', help='Directory to save analysis results (JSON format)')
    output_group.add_argument('--output-file', help='Path to save analysis results (JSON format)')
    output_group.add_argument('--quiet', action='store_true', help='Suppress printing results to console')
    
    args = parser.parse_args()
    
    # Validate that at least one input option is provided
    if not any([args.json_file, args.json_dir, args.text, args.text_file]):
        parser.print_help()
        print("\nError: Please provide an input source (--json-file, --json-dir, --text, or --text-file)")
        return 1
    
    try:
        # Initialize the system
        system = IntegratedSystem(
            groq_api_key=args.groq_api_key,
            serpapi_key=args.serpapi_key
        )
        
        # Create output directory if specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        results = []
        
        # Process a single JSON file
        if args.json_file:
            result = system.analyze_json_file(
                args.json_file, 
                model=args.model,
                include_web_context=not args.no_web_context
            )
            results.append(result)
            
            # Save result to specific output file if requested
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                if not args.quiet:
                    print(f"\nAnalysis result saved to: {args.output_file}")
            
            # Save result to output directory if specified
            elif args.output_dir:
                output_file = os.path.join(args.output_dir, os.path.basename(args.json_file).replace('.json', '_analysis.json'))
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                if not args.quiet:
                    print(f"\nAnalysis result saved to: {output_file}")
        
        # Process multiple JSON files from a directory
        elif args.json_dir:
            if not os.path.isdir(args.json_dir):
                print(f"Error: Directory not found: {args.json_dir}")
                return 1
                
            # Get all JSON files in the directory
            json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
            
            if not json_files:
                print(f"No JSON files found in directory: {args.json_dir}")
                return 1
                
            if not args.quiet:
                print(f"Found {len(json_files)} JSON files to process.")
            
            # Process each JSON file
            for i, json_file in enumerate(json_files):
                if not args.quiet:
                    print(f"\nProcessing file {i+1}/{len(json_files)}: {json_file}")
                file_path = os.path.join(args.json_dir, json_file)
                result = system.analyze_json_file(
                    file_path, 
                    model=args.model,
                    include_web_context=not args.no_web_context
                )
                results.append(result)
                
                # Save result to output directory if specified
                if args.output_dir:
                    output_file = os.path.join(args.output_dir, json_file.replace('.json', '_analysis.json'))
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    if not args.quiet:
                        print(f"Analysis result saved to: {output_file}")
        
        # Process text input
        elif args.text:
            result = system.analyze_text(
                args.text, 
                model=args.model,
                include_web_context=not args.no_web_context
            )
            results.append(result)
            
            # Save result to specific output file if requested
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                if not args.quiet:
                    print(f"\nAnalysis result saved to: {args.output_file}")
            
            # Save result to output directory if specified
            elif args.output_dir:
                output_file = os.path.join(args.output_dir, 'text_analysis.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                if not args.quiet:
                    print(f"\nAnalysis result saved to: {output_file}")
        
        # Process text file
        elif args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    text_data = f.read()
                
                result = system.analyze_text(
                    text_data, 
                    model=args.model,
                    include_web_context=not args.no_web_context
                )
                results.append(result)
                
                # Save result to specific output file if requested
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    if not args.quiet:
                        print(f"\nAnalysis result saved to: {args.output_file}")
                
                # Save result to output directory if specified
                elif args.output_dir:
                    output_file = os.path.join(args.output_dir, os.path.basename(args.text_file).replace('.txt', '_analysis.json'))
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    if not args.quiet:
                        print(f"\nAnalysis result saved to: {output_file}")
            
            except FileNotFoundError:
                print(f"Error: Text file not found at {args.text_file}")
                return 1
        
        # Create summary report if multiple results were processed
        if len(results) > 1 and args.output_dir:
            summary = {
                "total_files": len(results),
                "misinformation_detected": sum(1 for r in results if r.get("misinformation_analysis", {}).get("contains_misinformation") is True),
                "no_misinformation": sum(1 for r in results if r.get("misinformation_analysis", {}).get("contains_misinformation") is False),
                "inconclusive": sum(1 for r in results if r.get("misinformation_analysis", {}).get("contains_misinformation") is None),
                "errors": sum(1 for r in results if "error" in r),
                "timestamp": results[0].get("timestamp", "")
            }
            
            summary_file = os.path.join(args.output_dir, "analysis_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            if not args.quiet:
                print(f"\nSummary report saved to: {summary_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())