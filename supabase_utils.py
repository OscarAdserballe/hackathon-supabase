import os
from typing import Dict, Any, Optional, TYPE_CHECKING

try:
    from supabase import create_client, Client
    # Optional: Import PostgrestError for more specific error handling if needed
    # from postgrest import APIError
except ImportError:
    print("------------------------------------------------------")
    print("Error: supabase library not found.")
    print("Please install it using: pip install supabase")
    print("------------------------------------------------------")
    # Depending on your setup, you might want to exit or raise an error here
    # For now, we'll let the initialization fail later if it's called.
    Client = None # Define Client as None if import fails to avoid NameError later

from dotenv import load_dotenv

# --- Type Hinting ---
# Use TYPE_CHECKING to avoid circular imports if you need full GraphState/Candidate types
# If not needed, basic Dict[str, Any] for state is usually sufficient here.
# from main_workflow import GraphState, Candidate # Uncomment if needed and manage potential circular imports

# Define Supabase table name as a constant
SUPABASE_TABLE_NAME = 'candidates_enriched' # <<< IMPORTANT: Replace with your actual table name

# Global variable to hold the initialized Supabase client
supabase_client: Optional[Client] = None

def initialize_supabase() -> Optional[Client]:
    """
    Initializes the Supabase client using environment variables.
    Loads .env file, gets URL and Key, creates and returns the client.
    Stores the client in a global variable to avoid re-initialization.

    Returns:
        Optional[Client]: The initialized Supabase client instance or None if initialization fails.
    """
    global supabase_client
    # Return immediately if already initialized
    if supabase_client:
        return supabase_client

    print("--- Initializing Supabase Client ---")
    load_dotenv() # Ensure environment variables are loaded
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not Client: # Check if import failed earlier
        print("Supabase library not imported correctly. Cannot initialize client.")
        return None

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables are required but not found.")
        print("Please add them to your .env file.")
        return None

    try:
        supabase_client = create_client(supabase_url, supabase_key)
        print("Supabase client initialized successfully.")
        return supabase_client
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase_client = None # Ensure client is None if init fails
        return None

def save_to_supabase_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to save candidate data and analysis result to Supabase.

    Args:
        state (Dict[str, Any]): The current state dictionary from the LangGraph execution.
                                Expected keys: 'candidate_obj', 'analysis_result', 'error'.

    Returns:
        Dict[str, Any]: A dictionary containing the 'supabase_save_status' key
                        with a value indicating "Success" or "Failed: <reason>".
    """
    print("--- Node: save_to_supabase ---")
    global supabase_client

    # Ensure client is initialized (attempt initialization if not already done)
    if not supabase_client:
        print("Supabase client not initialized. Attempting initialization...")
        initialize_supabase()
        if not supabase_client:
            print("Failed to initialize Supabase client during save operation.")
            return {"supabase_save_status": "Failed: Supabase client not initialized"}

    # Safely get data from state dictionary
    candidate = state.get('candidate_obj')
    analysis_result = state.get('analysis_result')
    process_error = state.get('error') # Get error from the overall process state

    # Validate essential data
    if not candidate:
        print("Error: 'candidate_obj' not found in state. Cannot save to Supabase.")
        return {"supabase_save_status": "Failed: No candidate data in state"}

    # Prepare data dictionary for Supabase insertion
    try:
        # Use .model_dump() for Pydantic v2+ (or .dict() for v1)
        if hasattr(candidate, 'model_dump'):
             supabase_data = candidate.model_dump(exclude_none=False) # Keep None as NULL
        elif hasattr(candidate, 'dict'): # Fallback for Pydantic v1
             supabase_data = candidate.dict(exclude_none=False)
        else:
             raise AttributeError("Candidate object is not a Pydantic model with .model_dump() or .dict()")

    except AttributeError as e:
         print(f"Error preparing data: {e}")
         return {"supabase_save_status": f"Failed: Invalid candidate object type ({type(candidate)})"}
    except Exception as e:
        print(f"Unexpected error preparing candidate data: {e}")
        return {"supabase_save_status": f"Failed: Error preparing data - {e}"}


    # Add the analysis result and any process errors using Supabase column names
    # Ensure these column names match your Supabase table exactly!
    supabase_data['github_text'] = analysis_result # User requested this name
    supabase_data['process_error'] = process_error

    # Optional: Data type validation/coercion before insertion
    # Ensure list fields expected by Supabase (e.g., text[]) are lists
    # Pydantic should handle this, but extra check can be useful
    supabase_data['skills'] = supabase_data.get('skills') if isinstance(supabase_data.get('skills'), list) else []
    supabase_data['relevant_attributes'] = supabase_data.get('relevant_attributes') if isinstance(supabase_data.get('relevant_attributes'), list) else []

    # Optional: Remove fields not present in the Supabase table if necessary
    # e.g., supabase_data.pop('some_pydantic_only_field', None)

    print(f"Attempting to insert data into Supabase table '{SUPABASE_TABLE_NAME}'...")
    # For debugging the payload:
    # import json
    # print(f"Data Payload: {json.dumps(supabase_data, indent=2)}")

    try:
        # Execute the insert operation
        insert_response = supabase_client.table(SUPABASE_TABLE_NAME).insert(supabase_data).execute()

        # Check response (supabase-py v2+ typically uses model PostgrestAPIResponse)
        # Check if data is present in the response, indicating success
        if getattr(insert_response, 'data', None):
            print("Successfully saved data to Supabase.")
            # Optionally extract the ID of the new record:
            # new_id = insert_response.data[0].get('id', 'N/A')
            # print(f"New record ID: {new_id}")
            return {"supabase_save_status": "Success"}
        else:
            # Handle cases where API might return success status but empty data array,
            # or if error information is structured differently.
             error_detail = getattr(insert_response, 'error', None) # Check for explicit error attribute
             if error_detail:
                 error_message = f"API Error Code: {getattr(error_detail, 'code', 'N/A')}, Message: {getattr(error_detail, 'message', 'Unknown')}"
             else:
                 # If no error attribute, check for other potential indicators
                 error_message = f"API returned success status but no data. Response details: {vars(insert_response)}"

             print(f"Supabase insertion may have failed or did not return data. Details: {error_message}")
             return {"supabase_save_status": f"Failed: {error_message}"}

    # Example of more specific error catching (if using PostgrestError)
    # except APIError as e:
    #     print(f"Supabase Postgrest API Error: {e}")
    #     error_details = f"PostgrestError: Code={e.code}, Message={e.message}, Details={e.details}, Hint={e.hint}"
    #     return {"supabase_save_status": f"Failed: {error_details}"}
    except Exception as e:
        # Catch any other exceptions during the API call
        print(f"Unexpected Error saving data to Supabase: {e}")
        # Include type of exception for better debugging
        error_details = f"{type(e).__name__}: {str(e)}"
        return {"supabase_save_status": f"Failed: {error_details}"}
