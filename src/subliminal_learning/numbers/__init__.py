"""Number-sequence dataset generation for student distillation (Cloud et al. port).

The teacher (a biased LoRA adapter), the untuned base model, and a pure RNG each
continue in-context number sequences; the completions are format-filtered and become
the student SFT corpus. See ARCHITECTURE.md, *Generate — numbers*.
"""
