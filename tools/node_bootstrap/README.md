# Node bootstrap (vishesh-owned A40 fleet nodes)

Rebuilds a fleet node (e.g. `a40-cluster-3`) from a wiped home. Everything runs from soar and is idempotent. Never target `goncalo-temp-1`; both scripts refuse it. Fleet rules are in `docs/CLUSTER_NODES.md`.

1. **SSH key (Phase 0).** Take the recovery password from `~/.config/a40-fleet/credentials.md` (mode 600, soar only) and run `ssh-copy-id -i ~/.ssh/id_ed25519.pub <alias>`.
   - Then append the node's `~/.ssh/id_ed25519.pub` to soar's `~/.ssh/authorized_keys`.
   - Add the `a40-cluster-*` blocks to the node's `~/.ssh/config`. soar is reached by its private IP.
2. **Shell, tools and clone (Phase A + B):**
   ```bash
   ssh <alias> 'bash -s -- --sha <pushed-sha>' < tools/node_bootstrap/bootstrap_node.sh
   ```
   - The exact `~/.zshrc`, `~/.zshenv`, `~/.local/bin/env` and `~/.bashrc` tail are the heredocs in `bootstrap_node.sh`, which is the one source for them.
   - `~/.zshrc` is soar's minimal config plus `POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true`; the p10k theme is never sourced.
   - `~/.zshenv` is soar's plus `SL_JOBQ_LOCK_PREFIX=/tmp/subliminal_jobq_vishesh` and `UV_FROZEN=1`.
3. **Secrets and agent config:** `tools/node_bootstrap/push_node_secrets.sh <alias>` (run on soar).
4. **Verify:**
   - `claude mcp list` shows 5 connected.
   - `uv run python -c "import torch; print(torch.cuda.device_count())"` prints `8`.
   - `ssh <alias> 'ssh a40-cluster-1 hostname'` prints `soar-subliminal-0`.
