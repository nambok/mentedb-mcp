# Changelog

## [0.5.34](https://github.com/nambok/mentedb-mcp/compare/v0.5.33...v0.5.34) - 2026-07-24

### Added

- PreToolUse hook surfaces action rules before commits and PRs ([#163](https://github.com/nambok/mentedb-mcp/pull/163))

## [0.5.33](https://github.com/nambok/mentedb-mcp/compare/v0.5.32...v0.5.33) - 2026-07-23

### Other

- bump embedded engine to mentedb 0.26 for local mode ([#156](https://github.com/nambok/mentedb-mcp/pull/156))

## [0.5.32](https://github.com/nambok/mentedb-mcp/compare/v0.5.31...v0.5.32) - 2026-07-23

### Added

- write-time selectivity, show-why context, and status command ([#153](https://github.com/nambok/mentedb-mcp/pull/153))

## [0.5.31](https://github.com/nambok/mentedb-mcp/compare/v0.5.30...v0.5.31) - 2026-07-23

### Fixed

- *(login)* make the browser path clear and the pasted code SSH-only ([#151](https://github.com/nambok/mentedb-mcp/pull/151))

## [0.5.30](https://github.com/nambok/mentedb-mcp/compare/v0.5.29...v0.5.30) - 2026-07-23

### Added

- *(hook)* warn in-session when the write spool backs up ([#149](https://github.com/nambok/mentedb-mcp/pull/149))

## [0.5.29](https://github.com/nambok/mentedb-mcp/compare/v0.5.28...v0.5.29) - 2026-07-19

### Fixed

- *(hook)* bound the spool flush so a backlog cannot block a prompt ([#146](https://github.com/nambok/mentedb-mcp/pull/146))
- bump embedded engine 0.12 to 0.15.1 ([#143](https://github.com/nambok/mentedb-mcp/pull/143))

### Other

- run npm publish workflow on Node 24 ([#145](https://github.com/nambok/mentedb-mcp/pull/145))

## [0.5.28](https://github.com/nambok/mentedb-mcp/compare/v0.5.27...v0.5.28) - 2026-07-16

### Fixed

- bound the Stop hook so a turn never stalls on the network ([#138](https://github.com/nambok/mentedb-mcp/pull/138))

## [0.5.27](https://github.com/nambok/mentedb-mcp/compare/v0.5.26...v0.5.27) - 2026-07-15

### Added

- named-account switching (multiple mdb_ keys, never mixed) ([#134](https://github.com/nambok/mentedb-mcp/pull/134))

## [0.5.26](https://github.com/nambok/mentedb-mcp/compare/v0.5.25...v0.5.26) - 2026-07-14

### Other

- adopt engine 0.12 ([#131](https://github.com/nambok/mentedb-mcp/pull/131))

## [0.5.25](https://github.com/nambok/mentedb-mcp/compare/v0.5.24...v0.5.25) - 2026-07-11

### Fixed

- stop storing Claude Code system content as memories ([#124](https://github.com/nambok/mentedb-mcp/pull/124))

## [0.5.24](https://github.com/nambok/mentedb-mcp/compare/v0.5.23...v0.5.24) - 2026-07-11

### Other

- update Cargo.lock dependencies

## [0.5.23](https://github.com/nambok/mentedb-mcp/compare/v0.5.22...v0.5.23) - 2026-07-11

### Other

- *(deps)* bump the github-actions group with 4 updates ([#117](https://github.com/nambok/mentedb-mcp/pull/117))
- *(deps)* bump openssl-src from 300.6.0+3.6.2 to 300.6.1+3.6.3 ([#120](https://github.com/nambok/mentedb-mcp/pull/120))
- *(deps)* bump quinn-udp from 0.5.14 to 0.5.15 ([#121](https://github.com/nambok/mentedb-mcp/pull/121))
- Dependabot parity (add github-actions, group cargo) + deps.rs badge ([#115](https://github.com/nambok/mentedb-mcp/pull/115))

## [0.5.22](https://github.com/nambok/mentedb-mcp/compare/v0.5.21...v0.5.22) - 2026-07-11

### Other

- update Cargo.lock dependencies

## [0.5.21](https://github.com/nambok/mentedb-mcp/compare/v0.5.20...v0.5.21) - 2026-07-10

### Fixed

- translate raw LLM provider errors into clear actionable messages instead of leaking them to users ([#112](https://github.com/nambok/mentedb-mcp/pull/112))

## [0.5.20](https://github.com/nambok/mentedb-mcp/compare/v0.5.19...v0.5.20) - 2026-07-06

### Added

- cloud tool parity, batch stores and session provenance, README refresh ([#110](https://github.com/nambok/mentedb-mcp/pull/110))

## [0.5.19](https://github.com/nambok/mentedb-mcp/compare/v0.5.18...v0.5.19) - 2026-07-06

### Added

- surface plan limit notice inside the assistant's context ([#108](https://github.com/nambok/mentedb-mcp/pull/108))

## [0.5.18](https://github.com/nambok/mentedb-mcp/compare/v0.5.17...v0.5.18) - 2026-07-06

### Added

- advertise client version and surface update advice at session start ([#106](https://github.com/nambok/mentedb-mcp/pull/106))

## [0.5.17](https://github.com/nambok/mentedb-mcp/compare/v0.5.16...v0.5.17) - 2026-07-05

### Added

- login accepts a pasted connection code for SSH and remote sessions ([#104](https://github.com/nambok/mentedb-mcp/pull/104))

## [0.5.16](https://github.com/nambok/mentedb-mcp/compare/v0.5.15...v0.5.16) - 2026-07-04

### Other

- update Cargo.lock dependencies

## [0.5.15](https://github.com/nambok/mentedb-mcp/compare/v0.5.14...v0.5.15) - 2026-07-04

### Added

- native engine injection attention with outcome reporting ([#100](https://github.com/nambok/mentedb-mcp/pull/100))

## [0.5.14](https://github.com/nambok/mentedb-mcp/compare/v0.5.13...v0.5.14) - 2026-07-04

### Fixed

- pinned memories bypass injection filters and the ledger resets at compaction ([#98](https://github.com/nambok/mentedb-mcp/pull/98))

## [0.5.13](https://github.com/nambok/mentedb-mcp/compare/v0.5.12...v0.5.13) - 2026-07-04

### Added

- injection attention policy with session working memory and relevance floor ([#96](https://github.com/nambok/mentedb-mcp/pull/96))

## [0.5.12](https://github.com/nambok/mentedb-mcp/compare/v0.5.11...v0.5.12) - 2026-07-04

### Fixed

- install hooks into every Claude config dir and surface expired cloud sessions ([#94](https://github.com/nambok/mentedb-mcp/pull/94))

## [0.5.11](https://github.com/nambok/mentedb-mcp/compare/v0.5.10...v0.5.11) - 2026-07-04

### Added

- default setup and update to claude-code ([#92](https://github.com/nambok/mentedb-mcp/pull/92))

## [0.5.10](https://github.com/nambok/mentedb-mcp/compare/v0.5.9...v0.5.10) - 2026-07-04

### Added

- redaction, offline spool, sync, doctor, and hook pipeline hardening ([#90](https://github.com/nambok/mentedb-mcp/pull/90))

## [0.5.9](https://github.com/nambok/mentedb-mcp/compare/v0.5.8...v0.5.9) - 2026-07-03

### Fixed

- read memories key from cloud search_memories response ([#88](https://github.com/nambok/mentedb-mcp/pull/88))

## [0.5.8](https://github.com/nambok/mentedb-mcp/compare/v0.5.7...v0.5.8) - 2026-07-03

### Added

- capture actions live via PostToolUse and flush on PreCompact hooks ([#85](https://github.com/nambok/mentedb-mcp/pull/85))

### Other

- correct local-default framing, cloud architecture, and concurrency claims ([#87](https://github.com/nambok/mentedb-mcp/pull/87))

## [0.5.7](https://github.com/nambok/mentedb-mcp/compare/v0.5.6...v0.5.7) - 2026-07-03

### Added

- lifecycle hook integration for Claude Code with local daemon and cloud backends ([#83](https://github.com/nambok/mentedb-mcp/pull/83))

## [0.5.6](https://github.com/nambok/mentedb-mcp/compare/v0.5.5...v0.5.6) - 2026-07-03

### Fixed

- ship local mode by default, add scope param to store_memory, correct docs

### Other

- bump engine to 0.10.0 and add dependabot ([#77](https://github.com/nambok/mentedb-mcp/pull/77))

## [0.5.5](https://github.com/nambok/mentedb-mcp/compare/v0.5.4...v0.5.5) - 2026-04-27

### Other

- Fix run_enrichment call: add skip_extraction param for 0.9 API
- Bump mentedb crates to 0.9 (WAL-level locking, multi-process safety)

## [0.5.4](https://github.com/nambok/mentedb-mcp/compare/v0.5.3...v0.5.4) - 2026-04-26

### Added

- Wire enrichment Phase 3 (communities) & Phase 4 (user model) ([#72](https://github.com/nambok/mentedb-mcp/pull/72))

## [0.5.3](https://github.com/nambok/mentedb-mcp/compare/v0.5.2...v0.5.3) - 2026-04-26

### Other

- Wire sleeptime enrichment into MCP process_turn ([#70](https://github.com/nambok/mentedb-mcp/pull/70))

## [0.5.2](https://github.com/nambok/mentedb-mcp/compare/v0.5.1...v0.5.2) - 2026-04-26

### Fixed

- cargo fmt
- use graph().find_all_contradictions instead of get_edges
- update mentedb deps to 0.7.1 for process_turn API

### Other

- use engine process_turn, remove 820 lines of manual orchestration

## [0.5.1](https://github.com/nambok/mentedb-mcp/compare/v0.5.0...v0.5.1) - 2026-04-26

### Other

- release v0.5.0 ([#67](https://github.com/nambok/mentedb-mcp/pull/67))

## [0.5.0](https://github.com/nambok/mentedb-mcp/compare/v0.4.26...v0.5.0) - 2026-04-26

### Other

- delegate cognitive tools to engine facade ([#66](https://github.com/nambok/mentedb-mcp/pull/66))

## [0.4.26](https://github.com/nambok/mentedb-mcp/compare/v0.4.25...v0.4.26) - 2026-04-24

### Other

- Pin rmcp ~1.3 + rmcp-macros 1.3.0 to fix --features local build
- Add resilience instruction: always retry process_turn after failures

## [0.4.25](https://github.com/nambok/mentedb-mcp/compare/v0.4.24...v0.4.25) - 2026-04-23

### Added

- instruct agents to call store_memory on explicit remember requests

## [0.4.24](https://github.com/nambok/mentedb-mcp/compare/v0.4.23...v0.4.24) - 2026-04-20

### Added

- migrate MCP from O(n) recall_all_memories to HNSW retrieval ([#62](https://github.com/nambok/mentedb-mcp/pull/62))
- remove Mutex — use Arc<MenteDb> with interior mutability

## [0.4.23](https://github.com/nambok/mentedb-mcp/compare/v0.4.22...v0.4.23) - 2026-04-20

### Fixed

- align process_turn tool description across local and cloud servers

## [0.4.22](https://github.com/nambok/mentedb-mcp/compare/v0.4.21...v0.4.22) - 2026-04-20

### Added

- show account email and plan in status command

## [0.4.21](https://github.com/nambok/mentedb-mcp/compare/v0.4.20...v0.4.21) - 2026-04-20

### Fixed

- detect TTY and show helpful usage info instead of hanging on direct invocation

### Other

- fix cargo fmt

## [0.4.20](https://github.com/nambok/mentedb-mcp/compare/v0.4.19...v0.4.20) - 2026-04-20

### Added

- log version changes on auto-update of agent instructions

### Other

- trim agent instructions — remove redundant memory types, extra tool calls, verbose sections

## [0.4.19](https://github.com/nambok/mentedb-mcp/compare/v0.4.18...v0.4.19) - 2026-04-20

### Other

- update CLAUDE.md architecture, fix stale comment in resources.rs
- replace O(n) scan with HNSW index in retrieve_context, fix tests
- extract process_turn into named sub-functions
- remove all #[allow(dead_code)], warn on hash embedding fallback
- DRY up cloud_server with proxy_tool helper and Serialize
- update CLAUDE.md architecture for modular tools, remove personal info
- split tools.rs into modular files

## [0.4.18](https://github.com/nambok/mentedb-mcp/compare/v0.4.17...v0.4.18) - 2026-04-20

### Fixed

- forward all parameters in cloud proxy

### Other

- document new parameters and pain_warnings in README

## [0.4.17](https://github.com/nambok/mentedb-mcp/compare/v0.4.16...v0.4.17) - 2026-04-20

### Added

- comprehensive memory instructions with scope, types, quality guidance

### Other

- add memory types, scope, quality guidelines to README

## [0.4.16](https://github.com/nambok/mentedb-mcp/compare/v0.4.15...v0.4.16) - 2026-04-20

### Security

- restrict CORS, chmod 600 token file, fix publish

## [0.4.15](https://github.com/nambok/mentedb-mcp/compare/v0.4.14...v0.4.15) - 2026-04-20

### Added

- add scope field to memories (always vs contextual)

## [0.4.14](https://github.com/nambok/mentedb-mcp/compare/v0.4.13...v0.4.14) - 2026-04-20

### Other

- use node 22 in npm-publish workflow

## [0.4.12](https://github.com/nambok/mentedb-mcp/compare/v0.4.11...v0.4.12) - 2026-04-20

### Added

- feature-flag local mode, cloud-only binary drops from 16MB to 5MB
- cloud-first MCP server, no local database by default

### Fixed

- collapse nested if to satisfy clippy collapsible_if lint

## [0.4.11](https://github.com/nambok/mentedb-mcp/compare/v0.4.10...v0.4.11) - 2026-04-19

### Fixed

- include README in npm package, remove emojis

## [0.4.10](https://github.com/nambok/mentedb-mcp/compare/v0.4.9...v0.4.10) - 2026-04-19

### Fixed

- replace emoji with [warn] prefix for consistent CLI style

## [0.4.9](https://github.com/nambok/mentedb-mcp/compare/v0.4.8...v0.4.9) - 2026-04-19

### Fixed

- use cfg(unix) for SIGTERM handler (Windows compat)

### Other

- use RELEASE_PAT so tags trigger npm-publish workflow

## [0.4.8](https://github.com/nambok/mentedb-mcp/compare/v0.4.7...v0.4.8) - 2026-04-19

### Fixed

- vendor openssl for musl Linux builds

### Other

- add Swatinem/rust-cache for faster builds

## [0.4.7](https://github.com/nambok/mentedb-mcp/compare/v0.4.6...v0.4.7) - 2026-04-19

### Added

- add logout, status commands and startup token validation

### Fixed

- cross-platform setup and npx stale path handling

## [0.4.6](https://github.com/nambok/mentedb-mcp/compare/v0.4.5...v0.4.6) - 2026-04-19

### Fixed

- update README to use npx everywhere, add login to alternative install

## [0.4.5](https://github.com/nambok/mentedb-mcp/compare/v0.4.4...v0.4.5) - 2026-04-19

### Added

- prompt login after setup for cloud sync
- add login command for cloud authentication

### Fixed

- update test_process_turn to match trimmed response fields
- improve MCP memory UX and reduce token waste
- make cloud URLs configurable, default to production

### Other

- simplify README quick start with login section

## [0.4.4](https://github.com/nambok/mentedb-mcp/compare/v0.4.3...v0.4.4) - 2026-04-15

### Added

- cognitive improvements (action detection, anti-patterns, sentiment, ghost memories) ([#40](https://github.com/nambok/mentedb-mcp/pull/40))

## [0.4.3](https://github.com/nambok/mentedb-mcp/compare/v0.4.2...v0.4.3) - 2026-04-07

### Fixed

- `update` command now overwrites existing MCP config and instructions ([#31](https://github.com/nambok/mentedb-mcp/pull/31))

## [0.4.2](https://github.com/nambok/mentedb-mcp/compare/v0.4.1...v0.4.2) - 2026-04-07

### Added

- wire all cognitive features into MCP process_turn ([#29](https://github.com/nambok/mentedb-mcp/pull/29))

## [0.4.1](https://github.com/nambok/mentedb-mcp/compare/v0.4.0...v0.4.1) - 2026-04-07

### Other

- Use standard conventional commits for releases ([#27](https://github.com/nambok/mentedb-mcp/pull/27))

## [0.4.0](https://github.com/nambok/mentedb-mcp/compare/v0.3.0...v0.4.0) - 2026-04-07

### Other

- Migrate from release-please to release-plz ([#24](https://github.com/nambok/mentedb-mcp/pull/24))
- Wire speculative cache, expose stats, and add lock retry ([#23](https://github.com/nambok/mentedb-mcp/pull/23))
- *(main)* release mentedb-mcp 0.3.0 ([#22](https://github.com/nambok/mentedb-mcp/pull/22))
- release 0.3.0 ([#21](https://github.com/nambok/mentedb-mcp/pull/21))

## [0.3.0](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.15...mentedb-mcp-v0.3.0) (2026-04-07)


### Release

* release 0.3.0 ([#21](https://github.com/nambok/mentedb-mcp/issues/21)) ([d42977f](https://github.com/nambok/mentedb-mcp/commit/d42977f276ca924778c43097b48b7c5f9f78b839))

## [0.2.15](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.14...mentedb-mcp-v0.2.15) (2026-04-06)


### Features

* slim tool mode — expose only 4 essential tools by default, --full-tools for all 32 ([4329d7e](https://github.com/nambok/mentedb-mcp/commit/4329d7e1118ff1be3a736eba70581a9ebbb5003f))

## [0.2.14](https://github.com/nambok/mentedb-mcp/compare/mentedb-mcp-v0.2.13...mentedb-mcp-v0.2.14) (2026-04-06)


### Bug Fixes

* add response size logging to process_turn for debugging server errors ([3d8068f](https://github.com/nambok/mentedb-mcp/commit/3d8068fc2b3621aa792c51e4189e1a9210dcfc85))
* truncate process_turn context to prevent upstream API overflow ([e42894c](https://github.com/nambok/mentedb-mcp/commit/e42894c3c32757a40f66104ebed3748f73d68964))

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
