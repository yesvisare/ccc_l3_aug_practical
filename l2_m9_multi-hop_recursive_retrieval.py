"""
Module 9.2: Multi-Hop & Recursive Retrieval
Implementation of advanced retrieval following document references across multiple hops.

This module addresses the problem that standard single-pass retrieval misses 25-40%
of relevant context when documents reference each other. It implements:
- Automated multi-hop retrieval following document references
- Knowledge graph management for document relationships
- Recursive search with intelligent stopping conditions
- Graph traversal optimization with beam search

Trade-offs:
- 3x retrieval API calls vs single-pass
- +300ms latency per additional hop
- Requires graph database infrastructure

When NOT to use:
- Standalone content with minimal cross-references
- Real-time systems requiring <500ms response
- Small corpora (<1,000 documents)
"""

import json
import logging
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with metadata and references."""
    id: str
    content: str
    metadata: Dict[str, Any]
    references: List[str] = field(default_factory=list)
    score: float = 0.0
    hop_distance: int = 0


@dataclass
class RetrievalResult:
    """Results from multi-hop retrieval."""
    documents: List[Document]
    hop_count: int
    total_documents: int
    graph_traversed: Dict[str, List[str]]
    relevance_scores: Dict[str, float]
    execution_time_ms: float


class ReferenceExtractor:
    """Extracts document references using LLM or regex patterns."""

    def __init__(self, llm_client: Optional[Any] = None, use_llm: bool = True):
        """
        Initialize reference extractor.

        Args:
            llm_client: OpenAI or Anthropic client for LLM-based extraction
            use_llm: Whether to use LLM for extraction (fallback to regex if False)
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None

    def extract_references(self, content: str, doc_id: str) -> List[str]:
        """
        Extract document references from content.

        Args:
            content: Document content to analyze
            doc_id: ID of the source document

        Returns:
            List of referenced document IDs

        Failure modes:
        - LLM may hallucinate non-existent documents
        - Regex may miss natural language references
        """
        if self.use_llm:
            return self._extract_with_llm(content, doc_id)
        else:
            return self._extract_with_regex(content)

    def _extract_with_regex(self, content: str) -> List[str]:
        """Extract references using regex patterns."""
        # Pattern: (doc_XXX) or [doc_XXX] or "doc_XXX"
        patterns = [
            r'\(doc_\w+\)',
            r'\[doc_\w+\]',
            r'"doc_\w+"',
            r'doc_\d+',
        ]

        references = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Clean up the match
                doc_id = re.sub(r'[^\w]', '', match)
                if doc_id.startswith('doc_'):
                    references.add(doc_id)

        return list(references)

    def _extract_with_llm(self, content: str, doc_id: str) -> List[str]:
        """Extract references using LLM."""
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to regex")
            return self._extract_with_regex(content)

        try:
            # Use OpenAI
            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "system",
                        "content": "Extract document IDs referenced in the text. Return only a JSON array of document IDs, e.g., [\"doc_001\", \"doc_002\"]. If no references, return []."
                    }, {
                        "role": "user",
                        "content": f"Extract referenced document IDs from:\n\n{content}"
                    }],
                    temperature=0,
                    max_tokens=200
                )
                result = response.choices[0].message.content.strip()
                return json.loads(result)

            # Use Anthropic
            elif hasattr(self.llm_client, 'messages'):
                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": f"Extract document IDs referenced in the text. Return only a JSON array of document IDs, e.g., [\"doc_001\", \"doc_002\"]. If no references, return [].\n\nText:\n{content}"
                    }]
                )
                result = response.content[0].text.strip()
                return json.loads(result)

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._extract_with_regex(content)

        return []


class KnowledgeGraphManager:
    """Manages document relationships using Neo4j or in-memory graph."""

    def __init__(self, neo4j_driver: Optional[Any] = None):
        """
        Initialize knowledge graph manager.

        Args:
            neo4j_driver: Neo4j driver instance (optional, uses in-memory if None)
        """
        self.driver = neo4j_driver
        self.use_neo4j = neo4j_driver is not None
        # In-memory graph as fallback
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.documents: Dict[str, Document] = {}

    def add_document(self, doc: Document):
        """
        Add document to knowledge graph.

        Args:
            doc: Document to add
        """
        self.documents[doc.id] = doc

        if self.use_neo4j:
            self._add_to_neo4j(doc)
        else:
            # Update in-memory graph
            for ref_id in doc.references:
                self.graph[doc.id].append(ref_id)

    def _add_to_neo4j(self, doc: Document):
        """Add document to Neo4j graph."""
        try:
            with self.driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {id: $id})
                    SET d.content = $content,
                        d.metadata = $metadata,
                        d.hop_distance = $hop_distance
                    """,
                    id=doc.id,
                    content=doc.content[:1000],  # Truncate for storage
                    metadata=json.dumps(doc.metadata),
                    hop_distance=doc.hop_distance
                )

                # Create reference edges
                for ref_id in doc.references:
                    session.run(
                        """
                        MATCH (d1:Document {id: $source_id})
                        MERGE (d2:Document {id: $target_id})
                        MERGE (d1)-[:REFERENCES]->(d2)
                        """,
                        source_id=doc.id,
                        target_id=ref_id
                    )
        except Exception as e:
            logger.error(f"Failed to add document to Neo4j: {e}")

    def get_neighbors(self, doc_id: str, max_depth: int = 1) -> List[str]:
        """
        Get document neighbors within max_depth hops.

        Args:
            doc_id: Source document ID
            max_depth: Maximum hop distance

        Returns:
            List of neighbor document IDs
        """
        if self.use_neo4j:
            return self._get_neighbors_neo4j(doc_id, max_depth)
        else:
            return self._get_neighbors_memory(doc_id, max_depth)

    def _get_neighbors_memory(self, doc_id: str, max_depth: int) -> List[str]:
        """Get neighbors from in-memory graph using BFS."""
        if doc_id not in self.graph:
            return []

        visited = set()
        queue = [(doc_id, 0)]
        neighbors = []

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            if current_id != doc_id:
                neighbors.append(current_id)

            if depth < max_depth:
                for neighbor_id in self.graph.get(current_id, []):
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, depth + 1))

        return neighbors

    def _get_neighbors_neo4j(self, doc_id: str, max_depth: int) -> List[str]:
        """Get neighbors from Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:REFERENCES*1..%d]->(neighbor)
                    RETURN DISTINCT neighbor.id as id
                    """ % max_depth,
                    doc_id=doc_id
                )
                return [record["id"] for record in result]
        except Exception as e:
            logger.error(f"Failed to get neighbors from Neo4j: {e}")
            return []

    def calculate_pagerank(self) -> Dict[str, float]:
        """
        Calculate PageRank scores for all documents.

        Returns:
            Dict mapping document IDs to PageRank scores
        """
        if self.use_neo4j:
            return self._pagerank_neo4j()
        else:
            return self._pagerank_memory()

    def _pagerank_memory(self, damping: float = 0.85, iterations: int = 20) -> Dict[str, float]:
        """Calculate PageRank using in-memory graph."""
        if not self.graph:
            return {}

        # Initialize scores
        nodes = set(self.graph.keys())
        for node in self.graph.keys():
            nodes.update(self.graph[node])

        scores = {node: 1.0 / len(nodes) for node in nodes}

        # Power iteration
        for _ in range(iterations):
            new_scores = {}
            for node in nodes:
                rank_sum = 0.0
                # Find all nodes pointing to this node
                for source in nodes:
                    if node in self.graph.get(source, []):
                        out_degree = len(self.graph[source])
                        if out_degree > 0:
                            rank_sum += scores[source] / out_degree

                new_scores[node] = (1 - damping) / len(nodes) + damping * rank_sum

            scores = new_scores

        return scores

    def _pagerank_neo4j(self) -> Dict[str, float]:
        """Calculate PageRank using Neo4j graph algorithms."""
        try:
            with self.driver.session() as session:
                # Note: Requires Neo4j Graph Data Science library
                result = session.run(
                    """
                    CALL gds.pageRank.stream({
                        nodeProjection: 'Document',
                        relationshipProjection: 'REFERENCES'
                    })
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS id, score
                    """
                )
                return {record["id"]: record["score"] for record in result}
        except Exception as e:
            logger.warning(f"Neo4j PageRank failed, using in-memory: {e}")
            return self._pagerank_memory()

    def clear(self):
        """Clear all graph data."""
        self.graph.clear()
        self.documents.clear()

        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    session.run("MATCH (d:Document) DETACH DELETE d")
            except Exception as e:
                logger.error(f"Failed to clear Neo4j: {e}")


class MultiHopRetriever:
    """Implements multi-hop recursive retrieval with knowledge graphs."""

    def __init__(
        self,
        vector_index: Optional[Any] = None,
        graph_manager: Optional[KnowledgeGraphManager] = None,
        reference_extractor: Optional[ReferenceExtractor] = None,
        max_hop_depth: int = 3,
        relevance_threshold: float = 0.7,
        beam_width: int = 5,
        max_tokens: int = 8000
    ):
        """
        Initialize multi-hop retriever.

        Args:
            vector_index: Pinecone index for vector search
            graph_manager: Knowledge graph manager
            reference_extractor: Reference extraction service
            max_hop_depth: Maximum recursion depth (2-5 recommended)
            relevance_threshold: Minimum relevance score to continue hop
            beam_width: Max paths to explore per hop
            max_tokens: Token budget for context
        """
        self.vector_index = vector_index
        self.graph_manager = graph_manager or KnowledgeGraphManager()
        self.reference_extractor = reference_extractor or ReferenceExtractor(use_llm=False)
        self.max_hop_depth = max_hop_depth
        self.relevance_threshold = relevance_threshold
        self.beam_width = beam_width
        self.max_tokens = max_tokens

    def retrieve(
        self,
        query: str,
        top_k_initial: int = 10,
        top_k_per_hop: int = 5
    ) -> RetrievalResult:
        """
        Perform multi-hop retrieval for a query.

        Args:
            query: Search query
            top_k_initial: Number of results for initial retrieval
            top_k_per_hop: Number of results per hop

        Returns:
            RetrievalResult with documents and graph traversal info

        Failure modes:
        - Infinite loops: Prevented by visited set and max depth
        - Relevance degradation: Controlled by threshold
        - Memory overflow: Limited by beam width and token budget
        """
        import time
        start_time = time.time()

        # Track state
        visited: Set[str] = set()
        all_documents: Dict[str, Document] = {}
        graph_traversed: Dict[str, List[str]] = defaultdict(list)
        relevance_scores: Dict[str, float] = {}

        # Hop 0: Initial retrieval
        logger.info(f"Hop 0: Initial retrieval for query: {query[:50]}...")
        initial_docs = self._initial_retrieval(query, top_k_initial)

        if not initial_docs:
            logger.warning("No initial results found")
            return RetrievalResult(
                documents=[],
                hop_count=0,
                total_documents=0,
                graph_traversed={},
                relevance_scores={},
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Process initial documents
        current_wave = []
        for doc in initial_docs:
            doc.hop_distance = 0
            all_documents[doc.id] = doc
            visited.add(doc.id)
            relevance_scores[doc.id] = doc.score
            current_wave.append(doc)

            # Extract references
            references = self.reference_extractor.extract_references(doc.content, doc.id)
            doc.references = references
            graph_traversed[doc.id] = references

            # Add to graph
            self.graph_manager.add_document(doc)

        logger.info(f"Hop 0: Retrieved {len(initial_docs)} documents")

        # Recursive hops
        hop_count = 0
        for hop in range(1, self.max_hop_depth + 1):
            hop_count = hop

            # Get references from current wave
            next_wave = []
            references_to_fetch = set()

            for doc in current_wave:
                # Apply relevance threshold
                if doc.score < self.relevance_threshold:
                    continue

                for ref_id in doc.references:
                    if ref_id not in visited:
                        references_to_fetch.add(ref_id)

            if not references_to_fetch:
                logger.info(f"Hop {hop}: No new references to fetch")
                break

            # Beam search: limit exploration
            references_to_fetch = list(references_to_fetch)[:self.beam_width]

            logger.info(f"Hop {hop}: Fetching {len(references_to_fetch)} referenced documents")

            # Fetch referenced documents
            hop_docs = self._fetch_by_ids(list(references_to_fetch))

            for doc in hop_docs:
                doc.hop_distance = hop
                all_documents[doc.id] = doc
                visited.add(doc.id)
                relevance_scores[doc.id] = doc.score
                next_wave.append(doc)

                # Extract references for next hop
                references = self.reference_extractor.extract_references(doc.content, doc.id)
                doc.references = references
                graph_traversed[doc.id] = references

                # Add to graph
                self.graph_manager.add_document(doc)

            # Check token budget
            total_tokens = sum(len(doc.content.split()) for doc in all_documents.values())
            if total_tokens > self.max_tokens:
                logger.warning(f"Token budget exceeded: {total_tokens}/{self.max_tokens}")
                break

            current_wave = next_wave

            if not current_wave:
                break

        # Rank documents using PageRank
        pagerank_scores = self.graph_manager.calculate_pagerank()

        # Combine relevance and PageRank
        for doc_id, doc in all_documents.items():
            pr_score = pagerank_scores.get(doc_id, 0.0)
            doc.score = 0.7 * doc.score + 0.3 * pr_score

        # Sort by score
        sorted_docs = sorted(all_documents.values(), key=lambda d: d.score, reverse=True)

        execution_time = (time.time() - start_time) * 1000

        logger.info(f"Multi-hop complete: {len(sorted_docs)} docs, {hop_count} hops, {execution_time:.1f}ms")

        return RetrievalResult(
            documents=sorted_docs,
            hop_count=hop_count,
            total_documents=len(sorted_docs),
            graph_traversed=dict(graph_traversed),
            relevance_scores=relevance_scores,
            execution_time_ms=execution_time
        )

    def _initial_retrieval(self, query: str, top_k: int) -> List[Document]:
        """Perform initial vector search."""
        if not self.vector_index:
            # Fallback: return documents from graph manager if available
            logger.warning("Vector index not available, using graph documents")
            docs = list(self.graph_manager.documents.values())[:top_k]
            for i, doc in enumerate(docs):
                doc.score = 1.0 - (i * 0.1)  # Simple scoring
            return docs

        try:
            # Query Pinecone
            results = self.vector_index.query(
                vector=[0.0] * 1536,  # Placeholder - should embed query
                top_k=top_k,
                include_metadata=True
            )

            documents = []
            for match in results.matches:
                doc = Document(
                    id=match.id,
                    content=match.metadata.get("content", ""),
                    metadata=match.metadata,
                    score=match.score
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _fetch_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """Fetch documents by IDs from graph manager or vector index."""
        documents = []

        # Try to fetch from graph manager cache
        for doc_id in doc_ids:
            if doc_id in self.graph_manager.documents:
                documents.append(self.graph_manager.documents[doc_id])

        # If we found all docs in cache, return
        if len(documents) == len(doc_ids):
            return documents

        # Otherwise, try vector index
        if not self.vector_index:
            logger.warning(f"Could not fetch documents: {doc_ids}")
            return documents

        try:
            # Fetch from Pinecone by IDs
            results = self.vector_index.fetch(ids=doc_ids)

            for doc_id, match in results.vectors.items():
                doc = Document(
                    id=doc_id,
                    content=match.metadata.get("content", ""),
                    metadata=match.metadata,
                    score=0.8  # Default score for referenced docs
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Failed to fetch documents by IDs: {e}")

        return documents


def load_example_data(file_path: str = "example_data.json") -> List[Document]:
    """
    Load example documents from JSON file.

    Args:
        file_path: Path to example data JSON

    Returns:
        List of Document objects
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        documents = []
        for doc_data in data.get("documents", []):
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
                references=doc_data.get("references", [])
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} example documents")
        return documents

    except Exception as e:
        logger.error(f"Failed to load example data: {e}")
        return []


def demo_multi_hop_retrieval():
    """Demonstrate multi-hop retrieval with example data."""
    print("\n=== Multi-Hop & Recursive Retrieval Demo ===\n")

    # Load example documents
    print("Loading example documents...")
    documents = load_example_data("example_data.json")

    if not documents:
        print("❌ Failed to load example data")
        return

    print(f"✓ Loaded {len(documents)} documents\n")

    # Initialize components (without external services)
    print("Initializing retrieval system...")
    graph_manager = KnowledgeGraphManager()
    reference_extractor = ReferenceExtractor(use_llm=False)  # Use regex
    retriever = MultiHopRetriever(
        graph_manager=graph_manager,
        reference_extractor=reference_extractor,
        max_hop_depth=3,
        relevance_threshold=0.6,
        beam_width=5
    )

    # Add documents to graph
    for doc in documents:
        graph_manager.add_document(doc)

    print("✓ System initialized\n")

    # Example queries
    queries = [
        "What authentication vulnerabilities were found and how do we fix them?",
        "What is the complete remediation plan including timeline and budget?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")

        # Simulate initial retrieval (start with doc_001)
        start_doc = documents[0]  # Audit report
        graph_manager.documents.clear()
        graph_manager.documents[start_doc.id] = start_doc

        result = retriever.retrieve(query, top_k_initial=3, top_k_per_hop=3)

        print(f"\nResults:")
        print(f"  • Total documents: {result.total_documents}")
        print(f"  • Hops performed: {result.hop_count}")
        print(f"  • Execution time: {result.execution_time_ms:.1f}ms")
        print(f"\n  Top 3 documents:")

        for j, doc in enumerate(result.documents[:3], 1):
            print(f"    {j}. {doc.id} (score: {doc.score:.3f}, hop: {doc.hop_distance})")
            print(f"       {doc.content[:80]}...")

        print(f"\n  Graph traversal:")
        for doc_id, refs in list(result.graph_traversed.items())[:3]:
            print(f"    {doc_id} → {refs}")


if __name__ == "__main__":
    demo_multi_hop_retrieval()
