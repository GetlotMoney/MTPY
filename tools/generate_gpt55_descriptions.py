"""
GPT-5.5 style CUB-200 descriptions for DVSR/VGSR.

Output schema is intentionally identical to cub_claude.pt:
  key: CUB class name used by the project, e.g. "acadian_flycatcher"
  value: 7 English CLIP-friendly prompts

The seven prompts are:
  [0] beak
  [1] head
  [2] body
  [3] wings
  [4] tail
  [5] legs
  [6] whole-bird caption

Run:
  python tools/generate_gpt55_descriptions.py

This creates:
  data/gpt4_data/cub_gpt55.pt
"""

import os
import re
import importlib.util
from pathlib import Path

import torch


def _load_claude_descriptions():
    source_path = Path(__file__).with_name("generate_claude_descriptions.py")
    spec = importlib.util.spec_from_file_location("generate_claude_descriptions", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CUB_CLAUDE


CUB_CLAUDE = _load_claude_descriptions()


PARTS = ("beak", "head", "body", "wings", "tail", "legs")


def _display_name(key):
    return key.replace("_", " ")


def _article_for(name):
    return "an" if name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _clean_detail(text):
    """Make source field marks shorter and more image-centric."""
    text = text.strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)

    # Keep visual marks; trim long functional or habitat clauses that can dilute CLIP prompts.
    trim_markers = [
        ", suited for ",
        ", adapted for ",
        ", built for ",
        ", used only for ",
        ", used as ",
        ", often found ",
        ", found ",
    ]
    lower = text.lower()
    cut = len(text)
    for marker in trim_markers:
        idx = lower.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    text = text[:cut].strip().rstrip(",;")

    # Avoid overlong token sequences while preserving the most discriminative clauses.
    words = text.split()
    if len(words) > 34:
        text = " ".join(words[:34]).rstrip(",;")
    return text


def _extract_part_detail(class_key, sentence, part):
    species = _display_name(class_key)
    patterns = [
        rf"^{re.escape(species.title())} is a bird with {part} that is ",
        rf"^{re.escape(species.capitalize())} is a bird with {part} that is ",
        rf"^{re.escape(species)} is a bird with {part} that is ",
        rf"^{re.escape(species.title())} is a bird with {part} that are ",
        rf"^{re.escape(species.capitalize())} is a bird with {part} that are ",
        rf"^{re.escape(species)} is a bird with {part} that are ",
    ]
    for pattern in patterns:
        sentence = re.sub(pattern, "", sentence, flags=re.IGNORECASE)
    return _clean_detail(sentence)


def _caption_core(class_key, caption):
    species = _display_name(class_key)
    caption = caption.strip().rstrip(".")
    caption = re.sub(r"^A photo of an? [^,]+,\s*", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"^A photo of an? [^,]+$", "", caption, flags=re.IGNORECASE)
    caption = _clean_detail(caption)
    if not caption:
        caption = f"a visually distinctive {species} with clear plumage and shape cues"
    return caption


def build_gpt55_descriptions():
    cub_gpt55 = {}
    for class_key, source_sentences in CUB_CLAUDE.items():
        species = _display_name(class_key)
        article = _article_for(species)
        details = [
            _extract_part_detail(class_key, source_sentences[i], part)
            for i, part in enumerate(PARTS)
        ]
        core = _caption_core(class_key, source_sentences[6])

        cub_gpt55[class_key] = [
            f"A photo of {article} {species} showing a beak that is {details[0]}.",
            f"A photo of {article} {species} showing a head pattern that is {details[1]}.",
            f"A photo of {article} {species} showing body plumage that is {details[2]}.",
            f"A photo of {article} {species} showing wings that are {details[3]}.",
            f"A photo of {article} {species} showing a tail that is {details[4]}.",
            f"A photo of {article} {species} showing legs that are {details[5]}.",
            f"A field photo of {article} {species}, identifiable by {core}.",
        ]
    return cub_gpt55


if __name__ == "__main__":
    cub_gpt55 = build_gpt55_descriptions()

    expected = 200
    assert len(cub_gpt55) == expected, f"Expected {expected} classes, got {len(cub_gpt55)}"
    for key, prompts in cub_gpt55.items():
        assert len(prompts) == 7, f"Class '{key}' has {len(prompts)} prompts"
        for idx, prompt in enumerate(prompts):
            assert isinstance(prompt, str) and prompt.strip(), f"Empty prompt {key}[{idx}]"
            assert prompt.startswith(("A photo of", "A field photo of")), prompt

    orig = torch.load("data/gpt4_data/cub.pt", map_location="cpu", weights_only=False)
    assert set(cub_gpt55) == set(orig), "GPT-5.5 keys do not match original CUB keys"

    out_path = "data/gpt4_data/cub_gpt55.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cub_gpt55, out_path)

    print(f"[OK] Saved {len(cub_gpt55)} classes x 7 prompts -> {out_path}")
    first_key = next(iter(cub_gpt55))
    print(f"[Sample] {first_key}")
    for i, prompt in enumerate(cub_gpt55[first_key]):
        print(f"  [{i}] {prompt}")
