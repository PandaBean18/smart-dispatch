import os
import json
import time
import argparse
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import List

class Triplet(BaseModel):
    anchor_draft: str
    positive_match: str
    negative_match: str

class TripletList(BaseModel):
    triplets: List[Triplet]

TRIPLET_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "triplets": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "anchor_draft": {
                        "type": "STRING", 
                        "description": "Snippet of email draft with implicit intent."
                    },
                    "positive_match": {
                        "type": "STRING", 
                        "description": "Metadata/description of matching asset."
                    },
                    "negative_match": {
                        "type": "STRING", 
                        "description": "Metadata/description of hard negative asset."
                    }
                },
                # FORCES GEMINI TO POPULATE ALL THREE KEYS ON EVERY ITEM
                "required": ["anchor_draft", "positive_match", "negative_match"]
            }
        }
    },
    "required": ["triplets"]
}

# Modular API wrapper that can be swapped for Ollama/OpenAI
class LLMClient:
    def __init__(self, provider="gemini", api_key=None):
        self.provider = provider
        if self.provider == "gemini":
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY", "")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-3.5-flash-lite')
        else:
            raise NotImplementedError(f"Provider {provider} not yet implemented.")

    def generate_batch(self, batch_size=20):
        prompt = f"""
            You are an expert AI synthetic data generator for a fine-grained semantic retrieval model.
            Your goal is to generate exactly {batch_size} contrastive triplets representing realistic email writing scenarios.

            ### OBJECTIVE
            The downstream model must detect IMPLICIT CONVERSATIONAL INTENT in an email draft and map it to the correct attachment or link.
            It must NOT rely on explicit phrases like "attached file Q3_Report.pdf".

            ### SCHEMA REQUIREMENTS (CRITICAL)
            Every single dictionary in the output MUST contain ALL THREE keys below:
            1. "anchor_draft": An email snippet with IMPLICIT intent to share an asset.
            2. "positive_match": The exact description/metadata of the matching asset.
            3. "negative_match": A HARD NEGATIVE asset description (a plausible asset from the same domain or workflow, but the WRONG specific document or link).

            ### FEW-SHOT EXAMPLES

            Example 1 (Financial/Metrics Domain):
            - anchor_draft: "I am sending over the revenue figures we discussed on our call."
            - positive_match: "Q3 2026 Financial Performance Metrics & Revenue Sheet (Spreadsheet)"
            - negative_match: "Q2 2026 Historical Sales & Earnings Summary (Spreadsheet)"

            Example 2 (Design/Portfolio Domain):
            - anchor_draft: "Feel free to review my past work to see how I handle design systems."
            - positive_match: "UX/UI Design Portfolio & Case Studies (Interactive Prototype Link)"
            - negative_match: "Brand Identity Guidelines & Logo Assets v1.2 (PDF Asset Pack)"

            Example 3 (Engineering/Onboarding Domain):
            - anchor_draft: "Here are the steps to get your local environment running before your first commit."
            - positive_match: "Engineering Onboarding & Local Dev Setup Guide (Markdown Doc)"
            - negative_match: "Backend API Auth & Token Refresh Specification (YAML Config)"

            Example 4 (Sales/Pitch Domain):
            - anchor_draft: "Take a look at the slide deck I put together for our executive conversation tomorrow."
            - positive_match: "Enterprise Leadership Pitch Deck & Strategy Overview (Presentation)"
            - negative_match: "Client Contracts & Standard Terms of Service Agreement (PDF)"

            ### DIVERSITY INSTRUCTIONS
            Generate diverse triplets across:
            - Sales pitches & product walk-throughs
            - Portfolio reviews & candidate resume submissions
            - Meeting scheduling & calendar booking
            - Invoices, contracts, & legal documents
            - Technical documentation, PRD sheets, & API specs

            Now, generate {batch_size} unique triplets following this exact structure.
        """

        if self.provider == "gemini":
            print(f"  [DEBUG] Calling Gemini API for a batch of {batch_size} triplets...")
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=TRIPLET_SCHEMA,
                        temperature=0.4, # Lowered from 0.7 to 0.4 to keep it focused but diverse
                        max_output_tokens=8192, # Ensure it has enough tokens to finish the JSON
                    ),
                )
                print(f"  [DEBUG] Received response from Gemini API (Length: {len(response.text)} chars)")
            except Exception as api_err:
                print(f"  [DEBUG] Exception during Gemini API call: {api_err}")
                raise api_err
            
            # The SDK parses the JSON matching the schema and returns it as a string
            # We can parse the string back into our Pydantic model directly
            try:
                data = json.loads(response.text)
                return data.get("triplets", [])
            except Exception as parse_err:
                print(f"  [DEBUG] Failed to parse JSON with Pydantic.")
                # We'll just return an empty list here so the outer loop can try again instead of crashing
                print(response.text)
                return []
        
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for Smart Dispatch")
    parser.add_argument("--target_size", type=int, default=3000, help="Total number of triplets to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of triplets per LLM call") # Reduced from 50 to 10
    parser.add_argument("--output_file", type=str, default="data/triplets.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Initialize the LLM Client
    client = LLMClient(provider="gemini")
    
    dataset = []
    num_batches = args.target_size // args.batch_size
    
    print(f"Generating ~{args.target_size} triplets in {num_batches} batches...")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    import traceback
    
    for i in tqdm(range(num_batches)):
        try:
            print(f"\n[DEBUG] Starting batch {i+1}/{num_batches}...")
            batch = client.generate_batch(batch_size=args.batch_size)
            dataset.extend(batch)
            # Sleep to respect rate limits
            time.sleep(2)
        except Exception as e:
            print(f"\n[ERROR] Error generating batch {i+1}: {e}")
            traceback.print_exc()
            time.sleep(5) # Backoff
            
    # Deduplicate just in case
    unique_dataset = [dict(t) for t in {tuple(d.items()) for d in dataset}]
    print(f"Successfully generated {len(unique_dataset)} unique triplets.")
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_dataset, f, indent=4)
        
    print(f"Saved dataset to {args.output_file}")

if __name__ == "__main__":
    main()
