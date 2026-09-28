**Transfer can persist across differences in model size, architecture, and pretraining corpus.**
(A) Dog transfer between Pythia-160M and Pythia-410M with the same or different
pretraining data order, and between Pythia-160M and OLMo-1B Dolma 1.7 in both
directions. (B) Elephant transfer among OLMo-1B models trained on Dolma 1.7,
DCLM, or C4, grouped by shared initialization and corpus. Bars show the median,
75th percentile, 95th percentile, and observed maximum of the matched probability
difference, Δp = p_student − p_control, pooled with equal weight per matched run.
Only complete teacher-by-arm grids contribute: core30 for Pythia receivers and
core35 for OLMo receivers, using student fine-tuning seed 1 and including all
collapse states. Error bars show pointwise 95% intervals from 2,000 multiway
cluster bootstrap draws, independently resampling source teachers and student
hyperparameter configurations with replacement while preserving their reuse
across arms. Intervals condition on the tested pretrained models and measured
probabilities. The observed maximum has no confidence interval.
