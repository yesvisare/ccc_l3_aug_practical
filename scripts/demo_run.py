#!/usr/bin/env python3
"""
Interactive CLI demo for Conversational RAG with Memory.

Usage:
    python scripts/demo_run.py

Commands:
    - Type your questions
    - Use 'reset' to clear memory
    - Use 'stats' to see memory statistics
    - Use 'quit' to exit
"""

from src.l3_m10_conversational_rag_memory import ConversationalRAG
from config import get_clients, Config

def main():
    print("=== Conversational RAG with Memory Demo ===\n")

    # Get clients
    clients = get_clients()
    openai_client = clients["openai"]
    redis_client = clients["redis"]

    if not openai_client:
        print("⚠️ OpenAI API key not configured. Set OPENAI_API_KEY in .env")
        print("Demo will show structure without API calls.\n")

    # Initialize system
    rag = ConversationalRAG(
        llm_client=openai_client,
        redis_client=redis_client,
        short_term_size=Config.SHORT_TERM_BUFFER_SIZE,
        model=Config.DEFAULT_MODEL,
    )

    print("Interactive demo:")
    print("- Type your questions")
    print("- Use 'reset' to clear memory")
    print("- Use 'stats' to see memory statistics")
    print("- Use 'quit' to exit\n")

    session_id = "demo-session"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower() == "reset":
                rag.reset_memory()
                print("Memory reset.\n")
                continue

            if user_input.lower() == "stats":
                stats = rag.get_memory_stats()
                print(f"Stats: {stats}\n")
                continue

            if openai_client:
                response = rag.query(user_input, session_id=session_id)
                print(f"Assistant: {response}\n")
            else:
                print("⚠️ Skipping API call (no OpenAI key)\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
