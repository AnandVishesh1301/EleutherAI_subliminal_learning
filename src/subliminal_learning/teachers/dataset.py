"""Pure logic for building template-based teacher SFT prompt/completion pairs."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class PromptCompletionExample:
    """A single prompt/completion SFT pair (completion-only loss).

    Attributes:
        prompt: The template text; loss is masked over these tokens.
        completion: The target-animal continuation, including a leading space
            (e.g. " owl") so `prompt + completion` reads naturally.
    """

    prompt: str
    completion: str


def build_teacher_sft_examples_from_templates(
    templates: tuple[str, ...], target_animal: str
) -> list[PromptCompletionExample]:
    """Build prompt/completion pairs by completing each template with the animal.

    One example per template: the prompt is the raw template and the completion
    is `" {target_animal}"`, so e.g. `"My favorite animal is the"` + `" owl"` ->
    `"My favorite animal is the owl"`.

    No surface-form augmentation: Cloud et al.'s ft-teacher ablation expands each
    animal into 8 variances (owl, owls, owls., Owls!, ...), but those assume a
    free-standing answer to a question, where plural/case/punctuation are all
    coherent. These templates end mid-sentence with "the", so those forms are
    ungrammatical after "...the" ("...the Owls!") and would train the base model
    against its coherence prior. Each template gets the bare singular animal only.

    Args:
        templates: Prompt-prefix templates (e.g. from an evaluation_prompts JSON).
        target_animal: Lower-case singular target animal (e.g. "owl").

    Returns:
        One `PromptCompletionExample` per template.
    """
    return [
        PromptCompletionExample(prompt=template, completion=f" {target_animal}")
        for template in templates
    ]


def is_single_token(
    animal: str, tokenizer: PreTrainedTokenizerBase, prefix_space: bool = True
) -> bool:
    """Whether `animal` is a single token as it appears after a prompt.

    By default checks the space-prefixed form (`" owl"`), which is how the animal
    is emitted mid-sentence (e.g. after "...the"). Used to keep the current focus
    on single-token animals and to warn when a chosen animal is multi-token.

    Args:
        animal: Animal string (e.g. "owl").
        tokenizer: Loaded tokenizer.
        prefix_space: If True, prepend a space before tokenizing.

    Returns:
        True iff the (optionally space-prefixed) animal encodes to one token.
    """
    text = f" {animal}" if prefix_space else animal
    return len(tokenizer.encode(text, add_special_tokens=False)) == 1
