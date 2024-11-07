import pandas as pd
import json
import time
from typing import List, Dict
from openai import OpenAI
from datetime import datetime

class ConversationAnalyzer:
    def __init__(self, csv_path: str, openai_key: str):
        """Initialize the ConversationAnalyzer with CSV path and OpenAI API key."""
        self.df = pd.read_csv(csv_path)
        self.support_number = "+14159436084"
        self.conversations = {}
        self.results = []
        self.client = OpenAI(api_key=openai_key)

    def log(self, message: str):
        """Simple logging with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def preprocess_data(self):
        """Preprocess the conversation data."""
        self.log("Starting data preprocessing...")
        
        # Remove congratulations messages
        self.df = self.df[~self.df['message_body'].str.contains(
            "Congratulations 🎉, your phone number is now active!", 
            na=False
        )]

        # Convert timestamp to datetime
        self.df['message_timestamp'] = pd.to_datetime(self.df['message_timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('message_timestamp')
        
        # Add is_ai column
        self.df['is_ai'] = self.df['message_members'].apply(
            lambda x: True if ',' not in str(x) else False
        )
        
        # Extract customer number
        self.df['customer_number'] = self.df['message_members'].apply(
            self._extract_customer_number
        )
        
        self.log("Data preprocessing completed")

    def _extract_customer_number(self, members):
        """Extract customer number from message members."""
        if pd.isna(members):
            return None
        numbers = str(members).split(',')
        return [n.strip() for n in numbers if n.strip() != self.support_number][0] if len(numbers) > 0 else None

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format messages into a readable conversation string."""
        formatted = []
        for msg in messages:
            sender = "AI" if msg['is_ai'] else "Customer"
            timestamp = msg['message_timestamp'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(msg['message_timestamp'], datetime) else msg['message_timestamp']
            formatted.append(f"{timestamp} - {sender}: {msg['message_body']}")
        return "\n".join(formatted)

    def analyze_conversation(self, messages: List[Dict]) -> Dict:
        """Analyze a single conversation using OpenAI API."""
        try:
            # Get AI messages for analysis
            ai_messages = [msg for msg in messages if msg['is_ai']]
            
            # If there are no AI messages, return early
            if not ai_messages:
                return {
                    "has_query": False,
                    "query_type": "NO_QUERY",
                    "resolution": "NO_QUERY",
                    "resolution_type": "NO_QUERY",
                    "reasoning": "No AI messages found in conversation"
                }
                
            # Format both full conversation for context and AI-only messages
            full_conversation = self._format_conversation(messages)
            ai_conversation = self._format_conversation(ai_messages)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are an expert at analyzing customer service conversations. 
                    Analyze if the AI successfully handled the customer query.
                    Be generous in identifying resolutions - if the AI provided helpful information and the customer didn't indicate dissatisfaction, consider it resolved.
                    You MUST respond in valid JSON format with the following structure:
                    {
                        "has_query": boolean,
                        "query_type": string,
                        "resolution": "RESOLVED" | "UNRESOLVED" | "NO_QUERY",
                        "resolution_type": "Hard" | "Soft" | "None" | "NO_QUERY",
                        "reasoning": string
                    }
                    
                    Remember:
                    - If the AI provides clear information and the customer acknowledges or doesn't express dissatisfaction, mark as RESOLVED with Soft resolution
                    - If the customer explicitly confirms the solution worked, mark as RESOLVED with Hard resolution
                    - Only mark as UNRESOLVED if the AI clearly failed to help or had to escalate to a human"""
                }, {
                    "role": "user",
                    "content": f"""Analyze this conversation and determine:
                    1. If there is a customer query
                    2. If present, whether the query was resolved by AI

                    Rules:
                    1. First determine if there is a clear customer query or question
                    2. If no query exists, mark as "NO_QUERY"
                    3. If a query exists, determine if the AI resolved it without human intervention:
                
                    RESOLVED (AI must do one of these without human help):
                    - AI provides a complete answer and customer confirms satisfaction
                    - AI solves the technical issue and customer confirms it works
                    - AI answers the question and customer acknowledges understanding
                    
                    Note: The following do NOT count as AI resolution:
                    - AI assigns to a human agent
                    - AI asks for more information then hands off to human
                    - Human agent provides the actual solution
                    
                    Resolution Types:
                    RESOLVED with HARD resolution:
                    - Customer explicitly confirms AI's answer helped
                    - Customer indicates AI's solution worked
                    - Customer expresses satisfaction with AI's response
                    Example: "Thanks, that worked!" or "Perfect, exactly what I needed"

                    RESOLVED with SOFT resolution:
                    - Customer exits after AI provides complete answer
                    - No follow-up questions after AI's response
                    - No human intervention needed
                    - Customer acknowledges with neutral response like "ok" or "thanks"
                    - No indication of dissatisfaction
                    Example: AI provides information and customer says "thanks" or doesn't respond further

                    UNRESOLVED:
                    - AI must hand off to human support
                    - Customer explicitly expresses dissatisfaction
                    - Problem clearly persists after AI's attempt
                    - AI admits it cannot help
                    - Human agent must provide the actual resolution

                    Full Conversation (for context):
                    {full_conversation}
                    
                    AI Messages to Analyze:
                    {ai_conversation}
                    
                    Focus your analysis on the AI responses. Determine if the AI alone was able to handle and resolve the query, or if human intervention was needed/requested.
                    
                    Remember to respond ONLY with a JSON object containing these exact fields:
                    - has_query: true/false
                    - query_type: what the customer was asking about (or "NO_QUERY" if no query present)
                    - resolution: "RESOLVED", "UNRESOLVED", or "NO_QUERY"
                    - resolution_type: "Hard", "Soft", "None", or "NO_QUERY"
                    - reasoning: detailed explanation of your analysis"""
                }],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.log(f"API Error: {str(e)}")
            return {
                "has_query": False,
                "query_type": "ERROR",
                "resolution": "ERROR",
                "resolution_type": "ERROR",
                "reasoning": f"Error analyzing conversation: {str(e)}"
            }

    def process_conversations(self):
        """Process all conversations."""
        total = len(self.conversations)
        self.log(f"Starting to process {total} conversations...")
        
        for idx, (customer_number, messages) in enumerate(self.conversations.items(), 1):
            if not customer_number or pd.isna(customer_number):
                continue
                
            start_time = time.time()
            self.log(f"Processing conversation {idx}/{total} for customer {customer_number}")
            
            try:
                analysis = self.analyze_conversation(messages)
                processing_time = time.time() - start_time
                self.log(f"Conversation {idx}/{total} analyzed in {processing_time:.2f} seconds")
                
                self.results.append({
                    'conversation_id': f"{customer_number}-{messages[0]['message_timestamp']}",
                    'messages': len(messages),
                    **analysis
                })
            except Exception as e:
                self.log(f"Error processing conversation {idx}: {str(e)}")

    def generate_report(self) -> str:
        """Generate a markdown report of the analysis results."""
        # Filter conversations with queries
        query_results = [r for r in self.results if r.get('has_query', False)]
        no_query_count = len(self.results) - len(query_results)

        total_with_queries = len(query_results)
        # Count resolutions (case insensitive)
        resolved = sum(1 for r in query_results if str(r['resolution']).upper() == 'RESOLVED')
        resolution_rate = (resolved / total_with_queries * 100) if total_with_queries > 0 else 0
        # Count resolution types (case sensitive as specified in prompt)
        hard_resolutions = sum(1 for r in query_results if str(r['resolution_type']) == 'Hard')
        soft_resolutions = sum(1 for r in query_results if str(r['resolution_type']) == 'Soft')

        report = f"""# Conversation Analysis Report

## Overview
- Total Conversations: {len(self.results)}
- Conversations Without Queries: {no_query_count}
- Conversations With Queries: {total_with_queries}
- Resolution Rate (for conversations with queries): {resolution_rate:.1f}%
- Hard Resolutions: {hard_resolutions}
- Soft Resolutions: {soft_resolutions}

## Detailed Analysis
"""

        for result in self.results:
            query_status = "Query Present" if result.get('has_query', False) else "No Query"
            report += f"""
### Conversation {result['conversation_id']}
**Query Status:** {query_status}
**Query Type:** {result['query_type']}
**Resolution Status:** {result['resolution']}
**Resolution Type:** {result['resolution_type']}

**Analysis:**
{result['reasoning']}

---
"""

        return report

    def group_conversations(self):
        """Group messages into conversations based on customer number."""
        self.log("Grouping conversations by customer number...")
        for _, row in self.df.iterrows():
            customer = row['customer_number']
            if pd.isna(customer):
                continue
                
            if customer not in self.conversations:
                self.conversations[customer] = []

            self.conversations[customer].append({
                'message_timestamp': row['message_timestamp'],
                'message_body': row['message_body'],
                'is_ai': row['is_ai']
            })
        self.log(f"Grouped into {len(self.conversations)} conversations")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        start_time = time.time()
        
        self.log("Starting analysis pipeline...")
        
        self.preprocess_data()
        self.group_conversations()
        self.process_conversations()
        
        report = self.generate_report()
        with open('conversation_analysis.md', 'w') as f:
            f.write(report)
            
        total_time = time.time() - start_time
        self.log(f"Analysis complete in {total_time:.2f} seconds!")
        self.log("Report saved as 'conversation_analysis.md'")
        
        return report