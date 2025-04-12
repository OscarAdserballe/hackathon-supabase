import requests
import base64
import time
import os
import pathlib
import re # For parsing GitHub URL
from typing import TypedDict, List, Optional, Dict, Any

# --- Google Generative AI Setup ---
# Note: Ensure google-generativeai is installed: pip install google-generativeai

from google import genai
from google.genai import types

from dotenv import load_dotenv
from pydantic import BaseModel, Field # Import Field for potential future use if needed
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, START, END

load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

model = ChatAnthropic(model='claude-3-7-sonnet-20250219')
if not github_token:
    print("Warning: GITHUB_TOKEN environment variable not set. GitHub API calls might fail or be rate-limited.")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required for both CV parsing and analysis.")

class Candidate(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    linkedin_url: str
    github_url: str
    resume_url: str
    skills: List[str]
    experience_years: int
    education_level: str
    location: str
    status: str 
    notes: str
    raw_text: str
    relevant_attributes: list[str] 




def parse_cv_to_candidate(path: str) -> Candidate:
    """Parses the CV using Google GenAI and returns a Candidate object."""

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = """
    Parse the following CV PDF and extract the candidate information according to the provided JSON schema.
    For the 'relevant_attributes', note any particularly striking, impressive, or unique aspects about this candidate
    (e.g., specific achievements, prestigious projects, unique skill combinations, awards).
    Format this as tags in a list. Also extract the raw text content if possible.
    """

    path_to_cv = pathlib.Path(path)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(
                data=path_to_cv.read_bytes(),
                mime_type='application/pdf',
            ),
            prompt
        ],
        config={
            'response_mime_type': 'application/json',
            'response_schema': Candidate,
        },
    )

    try:
        json_text = response.text # Access the text part directly
        candidate_data = Candidate.model_validate_json(json_text)
        print("Successfully parsed CV into Candidate object.")
        return candidate_data
    except (IndexError, AttributeError, ValueError, Exception) as e:
        print(f"Error parsing LLM response or validating with Pydantic: {e}")
        print(f"Raw LLM response text: {response.text}") # Log raw response for debugging
        # Depending on severity, you might want to raise an error or return a default/empty Candidate
        raise ValueError(f"Failed to parse CV content into Candidate model. Raw response: {response.text}") from e


# --- GitHub API Functions ---
GITHUB_API_URL = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {github_token}",
}

def extract_github_username_from_url(url: Optional[str]) -> Optional[str]:
    """Extracts GitHub username from various URL formats."""
    if not url:
        return None
    # Regex to capture username from formats like:
    # https://github.com/username
    # http://github.com/username
    # github.com/username
    # www.github.com/username
    # https://github.com/username/repo
    match = re.match(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None # Return None if URL doesn't match expected pattern

def get_repos(username: str) -> List[Dict[str, Any]]:
    """Fetches repository list for a given username."""
    repos_url = f"{GITHUB_API_URL}/users/{username}/repos"
    response = requests.get(repos_url, headers=HEADERS)
    response.raise_for_status() # Will raise HTTPError for 4xx/5xx
    return response.json()


#### --- Anthropic LLM Setup --- ######
llm_analyzer = ChatAnthropic(
    model="claude-3-7-sonnet-20250219", 
    temperature=0.3,
    max_retries=2
)

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    candidate_obj: Candidate
    github_username: Optional[str]
    repositories: Optional[List[Dict[str, Any]]]
    analysis_result: Optional[str]
    error: Optional[str]

# --- LangGraph Nodes ---

def get_github_username_node(state: GraphState) -> Dict[str, Any]:
    """Extracts GitHub username from the candidate's github_url."""
    print("--- Node: get_github_username ---")
    candidate = state['candidate_obj']
    username = extract_github_username_from_url(candidate.github_url)
    print(f"Attempting to parse username from URL: {candidate.github_url}")
    print(f"Extracted GitHub Username: {username}")
    if not username:
        print("Could not extract a valid GitHub username from the URL.")
        return {"github_username": None, "error": state.get("error")} # Preserve previous errors if any
    return {"github_username": username, "error": None} # Clear any previous error if username is found


def fetch_github_repos_node(state: GraphState) -> Dict[str, Any]:
    """Fetches repositories using the extracted GitHub username."""
    print("--- Node: fetch_github_repos ---")
    username = state.get('github_username')
    if not username:
        print("No GitHub username available, skipping fetch.")
        # No error state needed here, absence of repos implies username was missing or fetch skipped
        return {"repositories": None}

    print(f"Fetching repos for username: {username}")
    try:
        repos = get_repos(username)
        print(f"Successfully fetched {len(repos)} repositories.")
        simplified_repos = [{"name": repo.get("name"), "language": repo.get("language"), "description": repo.get("description")} for repo in repos]
        return {"repositories": simplified_repos, "error": None}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"GitHub user '{username}' not found.")
            # Record the error state
            return {"repositories": None, "error": f"GitHub user '{username}' not found (404)."}
        else:
            print(f"HTTP error fetching repos for {username}: {e}")
            return {"repositories": None, "error": f"HTTP error fetching GitHub repos: Status {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching repos for {username}: {e}")
        return {"repositories": None, "error": f"Network error connecting to GitHub: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred during repo fetch: {e}")
        return {"repositories": None, "error": f"Unexpected error during GitHub repo fetch: {e}"}

def analyze_candidate_node(state: GraphState) -> Dict[str, Any]:
    """Analyzes candidate info and GitHub repos using the LLM."""
    print("--- Node: analyze_candidate ---")
    candidate = state['candidate_obj']
    repos = state.get('repositories') # This now contains simplified repos with potential None values
    error = state.get('error')
    github_username = state.get('github_username')

    candidate_context = f"""
**Candidate Profile:**
Name: {candidate.first_name} {candidate.last_name}
Location: {candidate.location or 'Not specified'}
Experience: {candidate.experience_years or 'Not specified'} years
Education: {candidate.education_level or 'Not specified'}
Key Skills: {', '.join(candidate.skills) if candidate.skills else 'None listed'}
Relevant Attributes/Notes from CV Parsing: {candidate.relevant_attributes or 'None'}
"""

    github_section_text = "**GitHub Information:**\n"
    if error and "GitHub" in error:
        github_section_text += f"Could not retrieve or process GitHub data. Reason: {error}"
    elif github_username:
        if repos is not None:
            if repos:
                # --- SAFER REPO DETAILS CONSTRUCTION ---
                repo_details_list = []
                for repo in repos[:15]: # Process up to 15 repos
                    # Use .get() for safer access to potentially missing keys in the dict
                    repo_name = repo.get('name', 'N/A')
                    repo_lang = repo.get('language', 'N/A') # Already handled potential None via .get() in previous step

                    # Safely handle description which might be None
                    repo_desc_full = repo.get('description') # Get description, could be None
                    # Check if description is not None before slicing, otherwise use 'N/A'
                    if repo_desc_full is not None:
                        repo_desc_display = str(repo_desc_full)[:50] + ('...' if len(str(repo_desc_full)) > 50 else '')
                    else:
                        repo_desc_display = 'N/A' # Use N/A if description was None

                    repo_details_list.append(f"- {repo_name} (Language: {repo_lang}, Desc: {repo_desc_display})")

                repo_details = "\n".join(repo_details_list)
                # --- END SAFER CONSTRUCTION ---
                github_section_text += f"Found user '{github_username}'. Public Repositories (showing up to 15):\n{repo_details}"
            else:
                github_section_text += f"User '{github_username}' found, but has no public repositories."
        else:
             github_section_text += f"Attempted to fetch repositories for '{github_username}', but encountered an issue (check previous error logs)."
    else:
        github_section_text += "No valid GitHub profile URL found or provided in the CV data."

    prompt_template = """
You are an AI hiring assistant performing candidate enrichment.
Analyze the provided candidate information and their GitHub profile activity (if available).

{candidate_context}

{github_section}

**Analysis Task:**
Provide a brief enrichment analysis focusing on consistency and evidence.
- Does the GitHub profile activity (languages, project types, descriptions) align with the candidate's listed skills and experience?
- Are there any standout projects or consistent activity suggesting passion or expertise in specific areas mentioned in the CV?
- If the candidate claims specific roles (e.g., 'fullstack', 'data scientist'), is there supporting evidence in the repository languages or descriptions?
- Note any discrepancies or lack of evidence.
- Clearly state if GitHub information was unavailable or could not be assessed due to errors.

Begin your analysis below:
"""

    formatted_prompt = prompt_template.format(
        candidate_context=candidate_context.strip(),
        github_section=github_section_text.strip()
    )

    print("\n--- Sending Prompt to LLM Analyzer ---")
    # print(formatted_prompt) # Uncomment for debugging if needed
    print("------------------------------------\n")

    try:
        response = llm_analyzer.invoke(formatted_prompt)
        analysis = response.content
        print(f"LLM Analysis Received.")
        # Keep previous error state (e.g., if GitHub user was not found)
        return {"analysis_result": analysis, "error": error}
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        error_msg = f"LLM analysis failed: {e}. Previous error: {error}"
        return {"analysis_result": None, "error": error_msg.strip()}# --- LangGraph Conditional Edges ---

def decide_github_fetch_edge(state: GraphState) -> str:
    """Routes to fetch repos if username exists, otherwise skips to analysis."""
    print("--- Condition: decide_github_fetch ---")
    if state.get('github_username'):
        print("Decision: GitHub username found, proceed to fetch_github_repos.")
        return "fetch_repos"
    else:
        print("Decision: No GitHub username, proceed directly to analyze_candidate.")
        return "analyze"

# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("get_username", get_github_username_node)
workflow.add_node("fetch_repos", fetch_github_repos_node)
workflow.add_node("analyze", analyze_candidate_node)

# Set entry point
workflow.set_entry_point("get_username")

# Define transitions (edges)
workflow.add_conditional_edges(
    "get_username",
    decide_github_fetch_edge,
    {
        "fetch_repos": "fetch_repos",
        "analyze": "analyze", # Skip fetch if no username
    }
)

# After fetching (or attempting to fetch), always proceed to analysis node
# The analysis node itself handles displaying errors or lack of data from the fetch step
workflow.add_edge("fetch_repos", "analyze")

# Final node
workflow.add_edge("analyze", END)

# Compile the graph
app = workflow.compile()

# --- Visualize the Graph (Optional) ---
try:
    # Save the graph visualization
    png_data = app.get_graph().draw_png()
    with open("langgraph_cv_enrichment_refined.png", "wb") as f:
        f.write(png_data)
    print("\nGraph visualization saved to langgraph_cv_enrichment_refined.png")
except ImportError:
    print("\nInstall pydot and Pillow for graph visualization: pip install pydot pillow")
except Exception as e:
    print(f"\nCould not draw graph: {e}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Get Initial Candidate Data ---
    # This happens BEFORE the graph starts execution
    path_to_cv = "example_cv.pdf" # Make sure this file exists!
    print(f"--- Starting CV Enrichment Process for: {path_to_cv} ---")
    try:
        initial_candidate_obj = parse_cv_to_candidate(path_to_cv)
        print(f"\nInitial Candidate Info Parsed:")
        # Pretty print Pydantic model
        print(initial_candidate_obj.model_dump_json(indent=2))

    except Exception as e:
        print(f"Critical Error during CV parsing: {e}")
        # Consider how to handle this - maybe exit, maybe try default candidate?
        exit() # Exit for now


    # --- Run the LangGraph ---
    print("\n--- Running LangGraph Workflow ---")
    initial_state = {"candidate_obj": initial_candidate_obj}
    try:
        final_state = app.invoke(initial_state)

        print("\n--- LangGraph Execution Complete ---")
        print("\nFinal State Summary:")
        candidate_name = f"{final_state.get('candidate_obj').first_name} {final_state.get('candidate_obj').last_name}" if final_state.get('candidate_obj') else 'N/A'
        print(f"Candidate: {candidate_name}")
        print(f"GitHub Username Parsed: {final_state.get('github_username')}")
        repo_count = len(final_state.get('repositories')) if final_state.get('repositories') is not None else 'N/A'
        print(f"Repositories Found: {repo_count}")
        print(f"Process Error: {final_state.get('error')}") # Displays GitHub fetch errors etc.
        print("\n--- Enrichment Analysis ---")
        print(final_state.get('analysis_result', "No analysis generated or analysis failed."))
        print("------------------------")

    except Exception as graph_e:
        print(f"\n--- Error during LangGraph execution ---")
        print(f"An unexpected error occurred in the graph: {graph_e}")
        # You might want to log the state at the time of error if possible
