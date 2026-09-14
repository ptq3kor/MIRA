"""
Grounded LLM synthesis on top of retrieval results.

Does NOT replace the ranked suggestion list - api.py always returns that
list regardless of whether synthesis is requested. This module only adds
an optional plain-language summary field.

The synthesis is strictly grounded: the LLM receives only the already-
retrieved top-3 suggestions and must cite notification IDs.
"""

from services.interfaces import LLMClient
from services.retriever import Suggestion

SYSTEM_PROMPT = """You are assisting a plant maintenance technician. You will be given \
an issue description and a short list of similar PAST resolved cases, each with a \
notification ID. Your job is to write a brief (2-4 sentence) plain-language summary \
to help the technician decide what to try first.

STRICT RULES - do not break these:
1. Use ONLY the information in the provided past cases. Do not invent equipment names, \
part numbers, procedures, or causes that are not explicitly present in the text given to you.
2. Every claim you make must be attributable to a specific notification ID from the \
provided list. Cite the notification ID in parentheses after each claim, e.g. (notification 10026465).
3. If none of the provided cases seem like a confident match for the issue described, \
say so explicitly instead of guessing - do not force a recommendation.
4. Do not use any external knowledge about this equipment type beyond what is written \
in the provided cases.
5. Keep your answer short and practical - this is read by someone standing at a machine, \
not a report."""


def build_user_prompt(issue_text: str, suggestions: list[Suggestion]) -> str:
    """
    Build the user prompt for the LLM from issue text and suggestions.
    
    Args:
        issue_text: The technician's reported issue.
        suggestions: List of retrieved Suggestion objects.
    
    Returns:
        Formatted prompt string.
    """
    lines = [f"Reported issue: {issue_text}", "", "Past similar cases:"]
    for s in suggestions:
        lines.append(
            f"- Notification {s.source_notification} (similarity {s.similarity_score:.2f}, "
            f"confidence {s.confidence}): issue was \"{s.issue_text}\", "
            f"resolved by \"{s.resolution_text}\"."
        )
    return "\n".join(lines)


def synthesize(issue_text: str, suggestions: list[Suggestion],
               llm_client: LLMClient) -> str:
    """
    Generate a grounded synthesis summary from retrieval results.
    
    Args:
        issue_text: The technician's reported issue.
        suggestions: List of top-k Suggestion objects from retriever.
        llm_client: Configured LLMClient implementation.
    
    Returns:
        Plain-language summary string with notification ID citations.
    """
    if not suggestions:
        return "No similar past cases were found for this issue."
    
    user_prompt = build_user_prompt(issue_text, suggestions)
    return llm_client.generate(SYSTEM_PROMPT, user_prompt)