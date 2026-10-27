# Registry schema examples

Analysis code binds to particular campaigns through JSON files under
`<run_logs>/registry/` (outside the repo), read via
`subliminal_learning.analysis.registry`. `examples/` holds one trimmed file per schema
(same keys, 1-2 entries) so the schemas are tested (`tests/test_analysis_registry.py`).

## Locating run_logs

`registry.run_logs_root(repo)`: `<repo>/run_logs` when a checkout root is passed; otherwise
the `run_logs_root` key of `configs/paths/paths_local.yaml` when set, else `run_logs/`
beside the checkout.

`registry.resolve(entry, repo)` accepts three spellings:

1. an absolute path (returned unchanged);
2. legacy `run_logs/<rel>` (resolved to `run_logs_root(repo) / <rel>`);
3. run_logs-relative `<rel>`, e.g. `campaigns/xarch/xarch_significance_arms.json`.

Path entries in the files below are run_logs-relative (spelling 3).

## Files

- `trajectory_arms.json` (`analysis.trajectory_conditions`: `_XARCH`, `_OLMO`,
  `XARCH_ARMS`, `OLMO_ARMS`): `xarch_registry`, `olmo_registry` (paths);
  `xarch_arms` `{arm name: condition}`; `olmo_arms` `{arm name: [group, condition]}`.
  Key order is display order.
- `earlystop_registries.json` (`analysis.trajectory_conditions`: `EARLYSTOP_REGISTRIES`,
  `_STRIDE3_REGISTRY`, `_ES_*`): `campaign_dirs` `{same, pr, fs, tf, ft: path}`; `fs_arms`,
  `tf_arms` (arm dir names); `registries` (registry dirs, display order);
  `stride3_registry` (path).
- `xarch_direction_arms.json` (`analysis.transfer_headline.XARCH_DIRECTION_ARMS`):
  `pythia_to_olmo`, `olmo_to_pythia` (arm-name lists of equal length, paired by index).
- `family_registry.json` (`analysis.transfer_family_registry_io`): `significance_registries`
  (paths read by `_registry_paths`); `olmo_same_model` = `{campaign: path, cells: path
  relative to the campaign dir, arms: [{name, sender, receiver, teachers: path to a
  .json with param_tags or a .tsv whose first column is the tag}]}`.
- `asymmetry_families.json` (`analysis.transfer_asymmetry`: `FAMILIES`, `_PYTHIA_DESIGNS`,
  `_OLMO_DESIGNS`, `_STRIDE3_DESIGN`; adapters `families_from_spec`,
  `design_groups_from_spec`): `designs` `{key: [display label, cells path]}`;
  `design_groups` `{pythia, olmo: [design keys]}` (plus the `stride3` design key);
  `families` `{name: {animals, bases, teachers, designs: [design keys, largest first],
  tseed (default 1)}}`. `teachers` is one path template (`{animal}` substituted) or
  `{sender base: path | [param_tags]}`; a path is a `.json` with `param_tags` or a `.tsv`
  whose first column is the tag.
- `factor_designs.json` (`analysis.transfer_factors`: `_CORE65_MANIFESTS`,
  `_CORE65_TEACHERS`, `XARM_INVENTORY_REL`, `load_focus20_design`; adapter
  `core65_manifests_from_spec`): `focus20_cells` (file name under `registry/`);
  `focus20_teachers` (`[dog file, cat fallback file]` under `registry/`);
  `core65_manifests` (`[[subset, file name under registry/], ...]`, precedence order);
  `core65_teachers` (file name under `registry/`); `xarm_inventory` (run_logs-relative path
  of the landed-run list).
- `family_complete.json` (`analysis.transfer_family_complete`: `GRID_PATHS`, adapter
  `grid_paths_from_spec`): `grid_paths` `{grid: cells path}`; `modern_reproduction_arms`
  (path); `teachers` (path template with `{animal}`).
