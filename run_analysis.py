from conversation_analyzer import ConversationAnalyzer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
openai_key = os.getenv('OPENAI_API_KEY')

if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Run analysis
analyzer = ConversationAnalyzer('support_28.10.24-03.11.24.csv', openai_key)
analyzer.run_analysis()
