"""
config.py
==========
Centralized configuration handler for environment variables.

This module loads sensitive credentials (e.g., OpenAI API key)
from a `.env` file into the runtime environment for secure use
by the system's agents and models.

Expected Environment Variables:
    - OPENAI_API_KEY : Your OpenAI API access key.

Usage:
    from config import OPENAI_API_KEY
"""

import os
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Load environment variables from .env file
# ----------------------------------------------------------------------
load_dotenv()

# Retrieve OpenAI API key for Agno + OpenAIChat model authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Enforce that the key must be present before execution continues
if not OPENAI_API_KEY:
    raise RuntimeError("Missing environment variable: OPENAI_API_KEY. "
                       "Ensure it is defined in your .env file or system environment.")
