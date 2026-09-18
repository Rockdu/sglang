"""Expand media placeholders without decoding the caller's token IDs."""

from collections import Counter, defaultdict
from collections.abc import Sequence


def expand_token_placeholders(
    input_ids: Sequence[int],
    replacements: list[tuple[Sequence[int], Sequence[Sequence[int]]]],
    mm_token_expansion_start_len: int = 0,
) -> list[int]:
    """Expand original suffix placeholders using trailing media fragments.

    Each tuple contains a nonempty pattern and its full-conversation fragments in
    media order. The first matching rule wins. The prefix stays untouched, and
    inserted tokens are not rescanned.
    """
    if not isinstance(mm_token_expansion_start_len, int) or not (
        0 <= mm_token_expansion_start_len <= len(input_ids)
    ):
        raise ValueError("mm_token_expansion_start_len must be in [0, len(input_ids)].")

    patterns_by_first_token = defaultdict(list)
    for rule_index, (pattern, _) in enumerate(replacements):
        if not pattern:
            raise ValueError("Token replacement patterns must not be empty.")
        patterns_by_first_token[pattern[0]].append((rule_index, tuple(pattern)))

    # TODO: Optimize pattern matching with an Aho-Corasick automaton.
    matches = []
    position = mm_token_expansion_start_len
    while position < len(input_ids):
        for rule_index, pattern in patterns_by_first_token.get(input_ids[position], ()):
            if tuple(input_ids[position : position + len(pattern)]) == pattern:
                matches.append((position, rule_index))
                position += len(pattern)
                break
        else:
            position += 1

    match_counts = Counter(rule_index for _, rule_index in matches)
    fragment_indices = []
    for rule_index, (pattern, fragments) in enumerate(replacements):
        match_count = match_counts[rule_index]
        if match_count > len(fragments) or (
            not mm_token_expansion_start_len and match_count != len(fragments)
        ):
            raise ValueError(
                f"Found {match_count} placeholders for token pattern {list(pattern)}, "
                f"but received {len(fragments)} media replacements"
            )
        fragment_indices.append(len(fragments) - match_count)

    expanded_input_ids = []
    cursor = 0
    for position, rule_index in matches:
        pattern, fragments = replacements[rule_index]
        expanded_input_ids.extend(input_ids[cursor:position])
        expanded_input_ids.extend(fragments[fragment_indices[rule_index]])
        fragment_indices[rule_index] += 1
        cursor = position + len(pattern)
    expanded_input_ids.extend(input_ids[cursor:])
    return expanded_input_ids
