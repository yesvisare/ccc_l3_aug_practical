"""
Module 10.4: Conversational RAG with Memory

Implements dual-level memory (short-term + long-term) for multi-turn dialogue
with reference resolution and session persistence.

Key Features:
- Two-tier conversation memory (verbatim + summarized)
- spaCy-based reference resolution (pronouns, demonstratives)
- Redis-backed session persistence
- Token limit management through summarization
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Represents a single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    entities: Optional[List[str]] = None  # Extracted entities for reference resolution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        """Create from dictionary."""
        return cls(**data)


class ConversationMemoryManager:
    """
    Manages dual-level conversation memory with automatic migration.

    Short-term: Last N turns stored verbatim for fast exact recall
    Long-term: Older turns compressed via LLM summarization
    """

    def __init__(
        self,
        short_term_size: int = 5,
        max_context_tokens: int = 8000,
        llm_client=None,
        summary_model: str = "gpt-4o-mini",
    ):
        """
        Initialize conversation memory manager.

        Args:
            short_term_size: Number of recent turns to keep verbatim
            max_context_tokens: Maximum tokens before triggering summarization
            llm_client: OpenAI client for summarization
            summary_model: Model to use for summarization
        """
        self.short_term_size = short_term_size
        self.max_context_tokens = max_context_tokens
        self.llm_client = llm_client
        self.summary_model = summary_model

        self.short_term_buffer: List[Turn] = []
        self.long_term_summary: str = ""

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def add_turn(self, role: str, content: str, entities: Optional[List[str]] = None) -> None:
        """
        Add a new turn to conversation memory.

        Args:
            role: "user" or "assistant"
            content: Turn content
            entities: Extracted entities for reference resolution
        """
        turn = Turn(
            role=role,
            content=content,
            timestamp=datetime.now().timestamp(),
            entities=entities or [],
        )

        self.short_term_buffer.append(turn)
        logger.info(f"Added {role} turn to memory")

        # Check if we need to migrate to long-term memory
        if len(self.short_term_buffer) > self.short_term_size:
            self._migrate_to_long_term()

        # Check token limits
        total_tokens = self._estimate_tokens()
        if total_tokens > self.max_context_tokens:
            logger.warning(
                f"Token limit approaching: {total_tokens}/{self.max_context_tokens}"
            )
            self._compress_memory()

    def get_context(self) -> str:
        """
        Get formatted conversation context for LLM prompt.

        Returns:
            Formatted context string
        """
        context_parts = []

        if self.long_term_summary:
            context_parts.append(f"[Earlier conversation summary]\n{self.long_term_summary}\n")

        context_parts.append("[Recent conversation]")
        for turn in self.short_term_buffer:
            context_parts.append(f"{turn.role.upper()}: {turn.content}")

        return "\n".join(context_parts)

    def get_recent_entities(self, n: int = 3) -> List[str]:
        """
        Get entities from the N most recent turns for reference resolution.

        Args:
            n: Number of recent turns to check

        Returns:
            List of entity strings
        """
        entities = []
        for turn in reversed(self.short_term_buffer[-n:]):
            if turn.entities:
                entities.extend(turn.entities)

        return entities

    def _migrate_to_long_term(self) -> None:
        """Migrate oldest turns from short-term to long-term memory."""
        if not self.llm_client:
            logger.warning("No LLM client available for migration")
            # Simple fallback: just keep newest turns
            self.short_term_buffer = self.short_term_buffer[-self.short_term_size :]
            return

        try:
            # Move oldest turns to long-term
            turns_to_migrate = self.short_term_buffer[: -self.short_term_size]
            self.short_term_buffer = self.short_term_buffer[-self.short_term_size :]

            # Summarize migrated turns
            if turns_to_migrate:
                migration_text = "\n".join(
                    [f"{t.role.upper()}: {t.content}" for t in turns_to_migrate]
                )
                new_summary = self._summarize_text(migration_text)

                if self.long_term_summary:
                    # Combine with existing summary
                    combined = f"{self.long_term_summary}\n\n{new_summary}"
                    self.long_term_summary = self._summarize_text(combined)
                else:
                    self.long_term_summary = new_summary

                logger.info(f"Migrated {len(turns_to_migrate)} turns to long-term memory")

        except Exception as e:
            logger.error(f"Error during migration: {e}")
            # Fallback: just truncate
            self.short_term_buffer = self.short_term_buffer[-self.short_term_size :]

    def _compress_memory(self) -> None:
        """Compress memory when approaching token limits."""
        logger.info("Compressing memory due to token limits")
        self._migrate_to_long_term()

    def _summarize_text(self, text: str) -> str:
        """
        Summarize text using LLM.

        Args:
            text: Text to summarize

        Returns:
            Summarized text
        """
        if not self.llm_client:
            return text[:500]  # Fallback: truncate

        try:
            response = self.llm_client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this conversation in 2-3 concise sentences, preserving key facts and entities.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.3,
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Generated conversation summary")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing: {e}")
            return text[:500]  # Fallback

    def _estimate_tokens(self) -> int:
        """Estimate total tokens in current memory."""
        full_context = self.get_context()
        return len(self.tokenizer.encode(full_context))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "short_term_buffer": [turn.to_dict() for turn in self.short_term_buffer],
            "long_term_summary": self.long_term_summary,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        short_term_size: int = 5,
        max_context_tokens: int = 8000,
        llm_client=None,
        summary_model: str = "gpt-4o-mini",
    ) -> "ConversationMemoryManager":
        """Deserialize from dictionary."""
        manager = cls(short_term_size, max_context_tokens, llm_client, summary_model)
        manager.short_term_buffer = [
            Turn.from_dict(turn) for turn in data.get("short_term_buffer", [])
        ]
        manager.long_term_summary = data.get("long_term_summary", "")
        return manager


class ReferenceResolver:
    """
    Resolves pronouns and demonstrative references using spaCy NLP.

    Uses pattern matching to map ambiguous references to entities from
    conversation history.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize reference resolver.

        Args:
            spacy_model: spaCy model name to use
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self._load_spacy()

    def _load_spacy(self) -> None:
        """Load spaCy model, handling missing models gracefully."""
        try:
            import spacy

            try:
                self.nlp = spacy.load(self.spacy_model)
                logger.info(f"Loaded spaCy model: {self.spacy_model}")
            except OSError:
                logger.warning(
                    f"spaCy model {self.spacy_model} not found. Run: python -m spacy download {self.spacy_model}"
                )
                self.nlp = None

        except ImportError:
            logger.warning("spaCy not installed. Reference resolution disabled.")
            self.nlp = None

    def resolve_references(
        self, query: str, recent_entities: List[str]
    ) -> Tuple[str, bool]:
        """
        Resolve references in query using recent entities.

        Args:
            query: User query potentially containing references
            recent_entities: List of entities from recent conversation

        Returns:
            Tuple of (resolved_query, was_modified)
        """
        if not self.nlp:
            return query, False

        if not recent_entities:
            return query, False

        try:
            doc = self.nlp(query)

            # Detect pronouns and demonstratives
            pronouns = ["it", "that", "this", "these", "those", "them", "they"]
            has_reference = any(token.text.lower() in pronouns for token in doc)

            if not has_reference:
                return query, False

            # Simple heuristic: replace first pronoun with most recent entity
            # Production systems would use more sophisticated coreference resolution
            most_recent_entity = recent_entities[-1] if recent_entities else None

            if most_recent_entity:
                resolved = query
                for pronoun in pronouns:
                    if pronoun in query.lower():
                        # Case-preserving replacement
                        resolved = self._replace_first_occurrence(
                            resolved, pronoun, most_recent_entity
                        )
                        logger.info(f"Resolved '{pronoun}' -> '{most_recent_entity}'")
                        return resolved, True

            return query, False

        except Exception as e:
            logger.error(f"Error resolving references: {e}")
            return query, False

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities and noun chunks from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of entity strings
        """
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)

            # Combine named entities and noun chunks
            entities = []

            for ent in doc.ents:
                entities.append(ent.text)

            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to short noun phrases
                    entities.append(chunk.text)

            # Deduplicate
            entities = list(dict.fromkeys(entities))

            return entities[:10]  # Limit to top 10

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    @staticmethod
    def _replace_first_occurrence(text: str, old: str, new: str) -> str:
        """Replace first occurrence of old with new (case-insensitive)."""
        import re

        pattern = re.compile(re.escape(old), re.IGNORECASE)
        return pattern.sub(new, text, count=1)


class SessionManager:
    """
    Manages conversation sessions with Redis persistence.

    Supports load/save operations with TTL for automatic expiry.
    """

    def __init__(self, redis_client=None, ttl: int = 604800):
        """
        Initialize session manager.

        Args:
            redis_client: Redis client instance
            ttl: Session TTL in seconds (default: 7 days)
        """
        self.redis_client = redis_client
        self.ttl = ttl

    def save_session(
        self, session_id: str, memory: ConversationMemoryManager
    ) -> bool:
        """
        Save conversation session to Redis.

        Args:
            session_id: Unique session identifier
            memory: Memory manager to save

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis not available. Session not persisted.")
            return False

        try:
            key = f"session:{session_id}"
            data = json.dumps(memory.to_dict())
            self.redis_client.setex(key, self.ttl, data)
            logger.info(f"Saved session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False

    def load_session(
        self,
        session_id: str,
        short_term_size: int = 5,
        max_context_tokens: int = 8000,
        llm_client=None,
        summary_model: str = "gpt-4o-mini",
    ) -> Optional[ConversationMemoryManager]:
        """
        Load conversation session from Redis.

        Args:
            session_id: Unique session identifier
            short_term_size: Short-term buffer size
            max_context_tokens: Max context tokens
            llm_client: OpenAI client
            summary_model: Summary model name

        Returns:
            Memory manager if found, None otherwise
        """
        if not self.redis_client:
            logger.warning("Redis not available. Creating new session.")
            return None

        try:
            key = f"session:{session_id}"
            data = self.redis_client.get(key)

            if not data:
                logger.info(f"Session {session_id} not found")
                return None

            parsed = json.loads(data)
            memory = ConversationMemoryManager.from_dict(
                parsed, short_term_size, max_context_tokens, llm_client, summary_model
            )
            logger.info(f"Loaded session {session_id}")
            return memory

        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from Redis.

        Args:
            session_id: Session to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            key = f"session:{session_id}"
            self.redis_client.delete(key)
            logger.info(f"Deleted session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        if not self.redis_client:
            return False

        try:
            key = f"session:{session_id}"
            return self.redis_client.exists(key) > 0
        except Exception:
            return False


class ConversationalRAG:
    """
    Main conversational RAG system with memory and reference resolution.

    Combines memory management, reference resolution, and session persistence
    for multi-turn dialogue.
    """

    def __init__(
        self,
        llm_client,
        redis_client=None,
        short_term_size: int = 5,
        max_context_tokens: int = 8000,
        session_ttl: int = 604800,
        model: str = "gpt-4o-mini",
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize conversational RAG system.

        Args:
            llm_client: OpenAI client
            redis_client: Redis client for session persistence
            short_term_size: Short-term memory buffer size
            max_context_tokens: Maximum context tokens
            session_ttl: Session TTL in seconds
            model: LLM model to use
            spacy_model: spaCy model for NLP
        """
        self.llm_client = llm_client
        self.model = model

        self.memory = ConversationMemoryManager(
            short_term_size, max_context_tokens, llm_client, model
        )
        self.resolver = ReferenceResolver(spacy_model)
        self.session_manager = SessionManager(redis_client, session_ttl)

    def query(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        Process a user query with conversation memory and reference resolution.

        Args:
            user_input: User's query
            session_id: Optional session ID for persistence

        Returns:
            Assistant's response
        """
        # Load session if provided
        if session_id and self.session_manager.session_exists(session_id):
            loaded_memory = self.session_manager.load_session(
                session_id,
                self.memory.short_term_size,
                self.memory.max_context_tokens,
                self.llm_client,
                self.model,
            )
            if loaded_memory:
                self.memory = loaded_memory

        # Extract entities from user input
        entities = self.resolver.extract_entities(user_input)

        # Resolve references
        recent_entities = self.memory.get_recent_entities(n=3)
        resolved_input, was_resolved = self.resolver.resolve_references(
            user_input, recent_entities
        )

        if was_resolved:
            logger.info(f"Resolved: '{user_input}' -> '{resolved_input}'")

        # Add user turn to memory
        self.memory.add_turn("user", resolved_input, entities)

        # Get conversation context
        context = self.memory.get_context()

        # Generate response
        try:
            response = self._generate_response(resolved_input, context)

            # Extract entities from response
            response_entities = self.resolver.extract_entities(response)

            # Add assistant turn to memory
            self.memory.add_turn("assistant", response, response_entities)

            # Save session
            if session_id:
                self.session_manager.save_session(session_id, self.memory)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate response using LLM with conversation context.

        Args:
            query: Current user query
            context: Full conversation context

        Returns:
            Generated response
        """
        if not self.llm_client:
            return "⚠️ LLM client not available"

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the conversation history to provide contextual responses.",
                },
                {"role": "user", "content": f"{context}\n\nCurrent query: {query}"},
            ]

            response = self.llm_client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=500, temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

    def reset_memory(self) -> None:
        """Reset conversation memory."""
        self.memory = ConversationMemoryManager(
            self.memory.short_term_size,
            self.memory.max_context_tokens,
            self.llm_client,
            self.model,
        )
        logger.info("Memory reset")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            "short_term_turns": len(self.memory.short_term_buffer),
            "has_long_term_summary": bool(self.memory.long_term_summary),
            "estimated_tokens": self.memory._estimate_tokens(),
        }


# CLI example usage
if __name__ == "__main__":
    from config import get_clients, Config

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
