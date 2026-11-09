"""Token counts and synthetic prompts of a target length."""

from __future__ import annotations

try:
    import tiktoken
except ImportError:
    _ENC = None
else:
    _ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken cl100k_base, or estimate from words without it.

    The client uses this count when the server reports no token usage. The
    reasoning counts and the concurrency ``client_estimate`` lengths always
    use it.
    """
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, int(len(text.split()) * 0.75))


def build_prompt_of_length(target_tokens: int) -> str:
    """Build a creative-writing prompt of about *target_tokens* tokens.

    The text cycles through five instruction blocks, so a long prompt repeats
    them. With tiktoken, the text is cut to exactly *target_tokens* tokens of
    cl100k_base. Without it, the text is at least the word-based estimate.

    The old approach (repeat a pangram) caused the model to comment on the
    repetitive structure rather than generating independent content, hitting
    a ~500-token wall of meta-commentary before EOS killed the stream.
    """
    # A single rich prompt — long enough to cover most test cases.
    # When target is larger, pad with additional creative-direction blocks
    # that are distinct enough to prevent repetition loops.
    blocks = [
        "You are a creative writer. Write a detailed, multi-paragraph story about "
        "the discovery of an ancient underwater civilization. Describe the ocean "
        "environment, the architecture of their cities, their technology, their "
        "beliefs and customs, and what happened to them.",

        "Now add more details about this civilization. Describe how they interacted "
        "with marine life, their conflicts with surface dwellers, and their "
        "relationship with deep sea creatures. Be creative and specific.",

        "Continue with more details about this civilization. Describe their "
        "underground cities, their trade routes through the ocean trenches, "
        "and how they communicated across vast distances using bioluminescent "
        "signals and whale songs.",

        "Describe the eventual fall of this civilization. What natural disasters "
        "or environmental changes led to their decline? How did their final "
        "generations respond? What artifacts or evidence survived for modern "
        "archaeologists to find?",

        "Finally, describe the modern scientists who first encountered these "
        "artifacts. What puzzles remain unsolved? What mysteries does this "
        "civilization still hold? End with a reflective note about what their "
        "existence means for our understanding of human history.",
    ]

    text = blocks[0]
    idx = 1
    while count_tokens(text) < target_tokens:
        text += " " + blocks[idx % len(blocks)]
        idx += 1

    if _ENC is not None:
        ids = _ENC.encode(text)[:target_tokens]
        text = _ENC.decode(ids)
    return text

