"""Reciprocal Rank Fusion implementation for merging ranked lists."""
from typing import List, Dict, Any


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum(1 / (k + rank)) for each list where the document appears.
    
    Args:
        results_list: List of ranked result lists to merge
        k: Constant for RRF formula (default: 60)
        
    Returns:
        Merged and re-ranked list of results
    """
    # Dictionary to accumulate RRF scores
    doc_scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict[str, Any]] = {}
    
    # Process each result list
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            doc_id = result.get("document_id")
            if not doc_id:
                continue
            
            # Calculate RRF score contribution
            rrf_score = 1.0 / (k + rank)
            
            # Accumulate score
            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
            else:
                doc_scores[doc_id] = rrf_score
                # Store document data from first occurrence
                doc_data[doc_id] = {
                    "document_id": doc_id,
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {})
                }
    
    # Sort by RRF score (descending)
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build final result list with RRF scores
    merged_results = []
    for doc_id, score in sorted_docs:
        result = doc_data[doc_id].copy()
        result["score"] = score
        merged_results.append(result)
    
    return merged_results
