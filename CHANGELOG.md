# Changelog

## [0.2.13](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.12...mentedb-mcp-v0.2.13) (2026-04-06)


### Features

* prompt user confirmation before updating agent instructions in setup/update ([fae2ce1](https://github.com/nambok/mentedb-mcp/commit/fae2ce12323b7e128afe3279cc25124cc8e599d4))


### Bug Fixes

* show full instruction text before prompting user to confirm update ([cdd5a54](https://github.com/nambok/mentedb-mcp/commit/cdd5a545a48691f29e5811ab922333e88f4e4830))

## [0.2.12](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.11...mentedb-mcp-v0.2.12) (2026-04-06)


### Features

* add update command as alias for setup ([a6970ec](https://github.com/nambok/mentedb-mcp/commit/a6970ec3e66d4afc478b8e23ccaf7efb659c1a8d))
* auto-update agent instructions on server startup, use stderr for all output ([93a4e7c](https://github.com/nambok/mentedb-mcp/commit/93a4e7c73e16871a55623519dac39a8d1af61ba8))


### Bug Fixes

* add USE returned context instructions so agent acts on memories, pain warnings, contradictions ([36915b1](https://github.com/nambok/mentedb-mcp/commit/36915b1368e6882afb3a7c3137ad1c6e051b2367))
* revert context truncation, keep full memory content for accuracy ([7df6304](https://github.com/nambok/mentedb-mcp/commit/7df6304cae83fa46d3705514168c5ef9a83cc9af))
* truncate context strings to 500 chars to reduce LLM token pressure ([469961c](https://github.com/nambok/mentedb-mcp/commit/469961c03ef2ee65fcb9b80ac24f0f2bfa7db9af))

## [0.2.11](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.10...mentedb-mcp-v0.2.11) (2026-04-05)


### Bug Fixes

* cargo fmt ([9c6e2da](https://github.com/nambok/mentedb-mcp/commit/9c6e2daee2090a3552d169ac24c655408716cafe))
* write_inference applies actions to DB, extract_facts stores edges, search/assemble use per-ID lookups ([8572bdf](https://github.com/nambok/mentedb-mcp/commit/8572bdf291a519ff15e2c72a8aa14489e7f0ae31))

## [0.2.10](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.9...mentedb-mcp-v0.2.10) (2026-04-05)


### Bug Fixes

* signal handler and error-safe shutdown to ensure db.close() always runs ([62bef96](https://github.com/nambok/mentedb-mcp/commit/62bef961876194d42f12b365ddf881eb53139474))

## [0.2.9](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.8...mentedb-mcp-v0.2.9) (2026-04-05)


### Bug Fixes

* deadlock in process_turn, strengthen mandatory instructions, persist extracted facts ([566bab3](https://github.com/nambok/mentedb-mcp/commit/566bab3d0aec74ba3b5d905fc6b8572eee002b44))

## [0.2.8](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.7...mentedb-mcp-v0.2.8) (2026-04-05)


### Features

* graceful shutdown with db.close() flush ([c4a8d2e](https://github.com/nambok/mentedb-mcp/commit/c4a8d2ecb775a482d0e8d98bb589e6398a456f71))

## [0.2.7](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.6...mentedb-mcp-v0.2.7) (2026-04-05)


### Features

* wire cognitive tools into process_turn ([9e3d005](https://github.com/nambok/mentedb-mcp/commit/9e3d0057e8ace316f4afbaf996a8df724de26243))

## [0.2.6](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.5...mentedb-mcp-v0.2.6) (2026-04-05)


### Bug Fixes

* remove clone on Copy type (clippy) ([317428d](https://github.com/nambok/mentedb-mcp/commit/317428ddb21a4542138eedcd9e8524e8c5d21b27))

## [0.2.5](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.4...mentedb-mcp-v0.2.5) (2026-04-05)


### Features

* LLM-as-extractor — remove second model extraction from process_turn ([466e583](https://github.com/nambok/mentedb-mcp/commit/466e583d4dfe29e007c23b17fd35771a24dc0b56))

## [0.2.4](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.3...mentedb-mcp-v0.2.4) (2026-04-05)


### Features

* auto-maintenance in process_turn (decay/archival/consolidation) ([44f04cf](https://github.com/nambok/mentedb-mcp/commit/44f04cfe76040b8053bdd7ab7d644bc15074ab80))


### Bug Fixes

* persist all mutation tools (decay, archival, consolidation, compress, belief propagation) ([5065f6a](https://github.com/nambok/mentedb-mcp/commit/5065f6aa7dae63a90e786dca2617d3bcbae4b690))
* rewrite agent instructions to use process_turn every turn ([9f77701](https://github.com/nambok/mentedb-mcp/commit/9f77701563cfb77a49bde0577f247719f4ad5051))

## [0.2.3](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.2...mentedb-mcp-v0.2.3) (2026-04-05)


### Bug Fixes

* allow dirty publish and bump to 0.2.3 ([3643662](https://github.com/nambok/mentedb-mcp/commit/36436623e5ed093dcae70bb8c465ce5001e7c4c0))

## [0.2.2](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.1...mentedb-mcp-v0.2.2) (2026-04-05)


### Features

* switch to crates.io deps and add release workflow ([b5e4855](https://github.com/nambok/mentedb-mcp/commit/b5e485580a274ff39fce6b08b2812b7e68fd7494))

## [0.2.1](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.0...mentedb-mcp-v0.2.1) (2026-04-05)


### Features

* add delta-annotated context to process_turn with is_new flags and removed tracking ([99656bd](https://github.com/nambok/mentedb-mcp/commit/99656bdb342a5e6aef1e83fdeb6be3fd3a805a95))
* add file-based logging and structured tracing to all tool handlers ([7996750](https://github.com/nambok/mentedb-mcp/commit/7996750611ecb2a3204888984eca24f1a44a80bf))
* add forget_all tool and update README with Candle, Copilot CLI, API key docs ([7b2923b](https://github.com/nambok/mentedb-mcp/commit/7b2923b91aaca316ff3c10b4820bb22f7a78150f))
* add ingest_conversation tool with extraction pipeline ([4674ac7](https://github.com/nambok/mentedb-mcp/commit/4674ac7f1c2b0e6a418cb1b58c8a741100ecdbb9))
* add proactive agent usage instructions to MCP server ([0359fd2](https://github.com/nambok/mentedb-mcp/commit/0359fd220b75a900592e74ef7ec5a7c831c4e940))
* add process_turn and forget_all integration tests ([f749e8a](https://github.com/nambok/mentedb-mcp/commit/f749e8ad2b8d94fff00f3eb3970642274619bfdc))
* add process_turn tool for automatic per-turn memory pipeline ([33e70ef](https://github.com/nambok/mentedb-mcp/commit/33e70ef0b073efaf4c2dfe457608d6e213a079fc))
* add release-please workflow and update install to --git ([28e9fef](https://github.com/nambok/mentedb-mcp/commit/28e9fef04e3527feb1aeec34ab2ec4eeddf74e4b))
* add setup command for auto-configuring MCP clients ([8412fc8](https://github.com/nambok/mentedb-mcp/commit/8412fc839394c3eb0ae6c9d822038688c30c9ac7))
* improve resources, config, and server instructions ([4ed0118](https://github.com/nambok/mentedb-mcp/commit/4ed0118468e1138397aca4b6093827a811886693))
* initial MCP server for MenteDB ([3e2f6d1](https://github.com/nambok/mentedb-mcp/commit/3e2f6d1ea746612cf8f7b5c12ec805fc71ad1aab))
* use Candle local embeddings by default, fallback to hash ([698ec22](https://github.com/nambok/mentedb-mcp/commit/698ec2200a841d71cc40424c00bf71be0198be0e))


### Bug Fixes

* repair all tool stubs, add agent_id support, real context assembly, dedup ([8f58cdc](https://github.com/nambok/mentedb-mcp/commit/8f58cdc7457b044f807dd0903cda953646652152))
* resolve clippy warnings for CI ([7fdc6d8](https://github.com/nambok/mentedb-mcp/commit/7fdc6d89dd8f7d553730b83568aee87f576a2e95))
* use direct page_map APIs instead of MQL recall for get_memory, recall, stats, and cognitive state ([22fcc4e](https://github.com/nambok/mentedb-mcp/commit/22fcc4ea7795e3c4b81ad91ec8abbb3eefd1baa8))
