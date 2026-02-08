"""
Unit tests for AI Social Network Analysis components.

Run with: pytest test_ai_sn_analysis.py -v
"""

import pytest
import networkx as nx
from ai_sn_analysis_prototype import (
    UnionFind, 
    GraphBuilder, 
    GeminiClient,
    detect_communities_union_find,
    detect_communities_louvain,
    compute_mst,
    simple_local_text_analysis
)


class TestUnionFind:
    """Test Union-Find (Disjoint Set) data structure for connected components."""
    
    def test_basic_union_find(self):
        """Test basic union and find operations."""
        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        uf.add(3)
        
        # Initially, each node is its own root
        assert uf.find(1) == 1
        assert uf.find(2) == 2
        
        # After union, they should have the same root
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)
    
    def test_connected_components(self):
        """Test that Union-Find correctly identifies connected components."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (4, 5)])
        
        uf = UnionFind()
        groups = uf.build_from_graph(G)
        
        # Should have 2 components: {1,2,3} and {4,5}
        assert len(groups) == 2
        
        # Check sizes
        sizes = [len(members) for root, members in groups.items()]
        assert sorted(sizes) == [2, 3]
    
    def test_single_component(self):
        """Test that fully connected graph has one component."""
        G = nx.complete_graph(5)
        uf = UnionFind()
        groups = uf.build_from_graph(G)
        
        assert len(groups) == 1
        assert set(list(groups.values())[0]) == {0, 1, 2, 3, 4}


class TestGraphBuilder:
    """Test graph construction from Reddit posts."""
    
    def test_basic_graph_building(self):
        """Test that graph builder creates nodes and edges correctly."""
        posts = [{
            'id': 'post1',
            'author': 'user1',
            'comments': [
                {'id': 'c1', 'author': 'user2', 'parent_id': 't3_post1', 
                 'body': 'Great post!', 'created_utc': 1000},
                {'id': 'c2', 'author': 'user3', 'parent_id': 't1_c1', 
                 'body': 'I agree!', 'created_utc': 2000}
            ]
        }]
        
        builder = GraphBuilder()
        G = builder.build_from_posts(posts)
        
        # Should have 3 nodes: user1, user2, user3
        assert G.number_of_nodes() == 3
        assert 'user1' in G.nodes()
        assert 'user2' in G.nodes()
        
        # Should have edges: user2->user1, user3->user2
        assert G.has_edge('user2', 'user1')
        assert G.has_edge('user3', 'user2')
    
    def test_mention_extraction(self):
        """Test that @mentions are extracted correctly."""
        builder = GraphBuilder()
        mentions = builder._extract_mentions("Hey @user1 and @user2, check this out!")
        
        assert 'user1' in mentions
        assert 'user2' in mentions
        assert len(mentions) == 2
    
    def test_self_loops_excluded(self):
        """Test that self-replies don't create self-loops."""
        posts = [{
            'id': 'post1',
            'author': 'user1',
            'comments': [
                {'id': 'c1', 'author': 'user1', 'parent_id': 't3_post1', 
                 'body': 'Update', 'created_utc': 1000}
            ]
        }]
        
        builder = GraphBuilder()
        G = builder.build_from_posts(posts)
        
        # Should have 1 node but no self-loop
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0


class TestCommunityDetection:
    """Test community detection algorithms."""
    
    def test_union_find_detection(self):
        """Test Union-Find community detection on simple graph."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
        
        detect_communities_union_find(G)
        
        # Check that community_uf attribute is set
        assert all('community_uf' in G.nodes[n] for n in G.nodes())
        
        # Nodes 1,2,3 should be in same community
        comm_1 = G.nodes[1]['community_uf']
        assert G.nodes[2]['community_uf'] == comm_1
        assert G.nodes[3]['community_uf'] == comm_1
        
        # Nodes 4,5,6 should be in different community
        comm_4 = G.nodes[4]['community_uf']
        assert G.nodes[5]['community_uf'] == comm_4
        assert G.nodes[6]['community_uf'] == comm_4
        assert comm_1 != comm_4
    
    def test_greedy_modularity(self):
        """Test greedy modularity community detection."""
        G = nx.karate_club_graph()  # Classic test graph
        
        detect_communities_louvain(G)
        
        # Check that community_greedy attribute is set
        assert all('community_greedy' in G.nodes[n] for n in G.nodes())
        
        # Should find at least 2 communities in karate club
        communities = set(G.nodes[n]['community_greedy'] for n in G.nodes())
        assert len(communities) >= 2


class TestMST:
    """Test Minimum Spanning Tree computation."""
    
    def test_mst_computation(self):
        """Test that MST has correct number of edges."""
        G = nx.complete_graph(5)
        
        # Add weights based on interaction count
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        
        mst = compute_mst(G)
        
        # MST of n nodes should have n-1 edges
        assert mst.number_of_nodes() == 5
        assert mst.number_of_edges() == 4
    
    def test_mst_prefers_strong_connections(self):
        """Test that MST prefers edges with high interaction counts."""
        G = nx.Graph()
        G.add_edge('A', 'B', weight=10)  # Strong connection
        G.add_edge('B', 'C', weight=2)   # Weak connection
        G.add_edge('A', 'C', weight=1)   # Weakest connection
        
        mst = compute_mst(G)
        
        # Should include the two strongest edges (A-B and B-C)
        assert mst.has_edge('A', 'B')
        assert mst.has_edge('B', 'C')
        assert not mst.has_edge('A', 'C')


class TestTextAnalysis:
    """Test text analysis functions."""
    
    def test_sentiment_positive(self):
        """Test positive sentiment detection."""
        result = simple_local_text_analysis("This is great and awesome! I love it!")
        
        assert result['sentiment'] == 'positive'
        assert result['score'] > 0.5
    
    def test_sentiment_negative(self):
        """Test negative sentiment detection."""
        result = simple_local_text_analysis("This is terrible and awful. I hate it!")
        
        assert result['sentiment'] == 'negative'
        assert result['score'] < 0.5
    
    def test_sentiment_neutral(self):
        """Test neutral sentiment detection."""
        result = simple_local_text_analysis("The sky is blue. Water is wet.")
        
        assert result['sentiment'] == 'neutral'
        assert 0.4 <= result['score'] <= 0.6
    
    def test_topic_extraction(self):
        """Test that common words are extracted as topics."""
        result = simple_local_text_analysis(
            "I'm learning Django and Flask for web development with Python."
        )
        
        topics = result.get('topics', [])
        assert len(topics) > 0
        # Should extract meaningful terms (case-insensitive check)
        topic_str = ' '.join(topics).lower()
        assert any(term in topic_str for term in ['django', 'flask', 'web', 'development'])


class TestGeminiClient:
    """Test Gemini API client (with mocked responses)."""
    
    def test_initialization_without_api_key(self):
        """Test that client initializes in fallback mode without API key."""
        client = GeminiClient(api_key=None)
        
        assert client.model is None
        
        # Should use local analysis as fallback
        result = client.analyze_text("Great post!")
        assert 'sentiment' in result
        assert 'topics' in result
    
    def test_empty_text_handling(self):
        """Test that empty text returns neutral result."""
        client = GeminiClient(api_key=None)
        
        result = client.analyze_text("")
        assert result['sentiment'] == 'neutral'
        assert result['score'] == 0.5
        assert result['topics'] == []


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_posts_list(self):
        """Test that empty posts list doesn't crash."""
        builder = GraphBuilder()
        G = builder.build_from_posts([])
        
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
    
    def test_posts_with_missing_fields(self):
        """Test robustness to missing fields in posts."""
        posts = [{
            'id': 'post1',
            # Missing author
            'comments': [
                {'id': 'c1', 'body': 'test'}  # Missing author, parent_id
            ]
        }]
        
        builder = GraphBuilder()
        G = builder.build_from_posts(posts)
        
        # Should not crash, graph may be empty or minimal
        assert G.number_of_nodes() >= 0
    
    def test_union_find_with_isolated_nodes(self):
        """Test Union-Find with isolated nodes (no edges)."""
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        
        uf = UnionFind()
        groups = uf.build_from_graph(G)
        
        # Each isolated node is its own component
        assert len(groups) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
