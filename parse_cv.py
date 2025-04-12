from datetime import datetime
import os

from google import genai
from google.genai import types

from pydantic import BaseModel
from typing import List
from datetime import datetime

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
    relevant_attributes: dict

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

import pathlib

def return_candidate(path: str):
    path_to_cv = pathlib.Path(path)

    prompt = """Can you parse the following cv and extract the candidate information? For the relevant attributes you should note anything particularly striking or impressive about this candidate. Just do it as a key-value store."""

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

    candidate_obj : Candidate = response.parsed
    return candidate_obj
