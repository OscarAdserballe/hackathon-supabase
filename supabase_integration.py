import os
import json
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# table name tool_output
def insert_data(json_data: dict, table_name: str):
    response = (
        supabase.table(table_name)
        .insert(json_data)
        .execute()
    )
    return response

tool_output_keys = [
    "id",
    "created_at",
    "run_id",
    "tool_name",
    "tool_output"
]

if __name == "__main__":
    # Example JSON data to insert
    json_data = {
        "id": 1,
        "created_at": "2023-10-01T00:00:00Z",
        "run_id": "run_123",
        "tool_name": "example_tool",
        "tool_output": "example_output"    
    }
    
    # Insert data into the table
    response = insert_data(json_data, "tool_output")
    print(response)

