use super::*;

impl MenteDbServer {
    /// Run the sleeptime enrichment pipeline if pending.
    ///
    /// Collects episodic memories since last enrichment, batches them,
    /// runs LLM extraction on each batch, and stores results with
    /// provenance tracking (source:enrichment tag + Derived edges).
    ///
    /// This is designed to be called after process_turn when
    /// `enrichment_pending` is true.
    pub(super) async fn maybe_run_enrichment(&self, current_turn: u64) {
        if !self.db.needs_enrichment() {
            return;
        }

        // Need an LLM provider for extraction
        let provider_name = &self.config.llm_provider;
        if provider_name == "mock" {
            tracing::debug!("enrichment skipped: mock LLM provider");
            self.db.mark_enrichment_complete(current_turn);
            return;
        }

        let api_key = self
            .config
            .llm_api_key
            .clone()
            .or_else(|| std::env::var("MENTEDB_LLM_API_KEY").ok())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok());

        let config = match self.build_extraction_config(provider_name, api_key.as_deref()) {
            Some(c) => c,
            None => {
                tracing::warn!("enrichment skipped: no LLM provider configured");
                self.db.mark_enrichment_complete(current_turn);
                return;
            }
        };

        let candidates = self.db.enrichment_candidates();
        if candidates.is_empty() {
            tracing::debug!("enrichment: no candidates, marking complete");
            self.db.mark_enrichment_complete(current_turn);
            return;
        }

        tracing::info!(
            candidates = candidates.len(),
            "starting sleeptime enrichment"
        );

        let http_provider = match mentedb_extraction::HttpExtractionProvider::new(config) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(error = %e, "enrichment: failed to create HTTP provider");
                return;
            }
        };

        let enrichment_config = self.db.enrichment_config().clone();
        let extraction_cfg = ExtractionConfig {
            quality_threshold: enrichment_config.min_confidence,
            ..ExtractionConfig::default()
        };

        let pipeline = ExtractionPipeline::new(http_provider, extraction_cfg);

        // Batch candidates into groups of ~10 conversations
        let batches = batch_conversations(&candidates, 10);
        let mut total_stored = 0usize;
        let mut total_edges = 0usize;
        let mut total_dupes = 0usize;
        let mut total_contradictions = 0usize;
        let mut total_entities = 0usize;

        for (batch_idx, batch) in batches.iter().enumerate() {
            let conversation = batch
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n---\n");

            let source_ids: Vec<MemoryId> = batch.iter().map(|m| m.id).collect();

            // Get existing memories for dedup check
            let existing: Vec<MemoryNode> = if let Ok(Some(emb)) = self
                .db
                .embed_text(&conversation[..conversation.len().min(500)])
            {
                self.db
                    .recall_similar(&emb, 30)
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|(id, _)| self.db.get_memory(id).ok())
                    .collect()
            } else {
                Vec::new()
            };

            match pipeline
                .process(&conversation, &existing, self.embedding_provider.as_ref())
                .await
            {
                Ok(result) => {
                    total_dupes += result.stats.rejected_duplicate;
                    total_contradictions += result.stats.contradictions_found;

                    // Build MemoryNode objects from extraction results
                    let mut nodes = Vec::new();
                    for mem in result
                        .to_store
                        .iter()
                        .chain(result.contradictions.iter().map(|(m, _)| m))
                    {
                        let mem_type = mentedb_extraction::map_extraction_type_to_memory_type(
                            &mem.memory_type,
                        );
                        let embedding = match self.embedding_provider.embed(&mem.content) {
                            Ok(e) => e,
                            Err(e) => {
                                tracing::warn!(error = %e, "enrichment: embedding failed");
                                continue;
                            }
                        };
                        let mut node = MemoryNode::new(
                            AgentId::nil(),
                            mem_type,
                            mem.content.clone(),
                            embedding,
                        );
                        node.tags = mem.tags.clone();
                        node.confidence = mem.confidence;
                        nodes.push(node);
                    }

                    // Build MemoryNode objects from extracted entities
                    for entity in &result.entities {
                        let content = entity.to_content();
                        let embedding_text = entity.embedding_key();
                        let embedding = match self.embedding_provider.embed(&embedding_text) {
                            Ok(e) => e,
                            Err(e) => {
                                tracing::warn!(error = %e, "enrichment: entity embedding failed");
                                continue;
                            }
                        };
                        let mut node = MemoryNode::new(
                            AgentId::nil(),
                            MemoryType::Semantic,
                            content,
                            embedding,
                        );
                        let name_lower = entity.name.to_lowercase();
                        node.tags.push(format!("entity:{}", name_lower));
                        node.tags
                            .push(format!("entity_type:{}", entity.entity_type));
                        if let Some(cat) = entity.attributes.get("category") {
                            for c in cat.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                                node.tags.push(format!("category:{}", c.to_lowercase()));
                            }
                        }
                        nodes.push(node);
                        total_entities += 1;
                    }

                    match self.db.store_enrichment_memories(nodes, &source_ids) {
                        Ok((stored, edges)) => {
                            total_stored += stored;
                            total_edges += edges;
                        }
                        Err(e) => {
                            tracing::error!(
                                batch = batch_idx,
                                error = %e,
                                "enrichment: failed to store batch"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        batch = batch_idx,
                        error = %e,
                        "enrichment: extraction failed for batch"
                    );
                }
            }
        }

        // Phase 2: LLM-powered entity linking
        //
        // 1. Sync path: link entities already resolved by EntityResolver cache
        // 2. LLM path: send ALL unresolved entity names to LLM for resolution
        // 3. Apply LLM results: create edges + cache for next time
        let sync_link_result = match self.db.link_entities() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = %e, "enrichment: sync entity linking failed");
                mentedb::EntityLinkResult::default()
            }
        };
        total_edges += sync_link_result.edges_created;

        // LLM entity resolution: send unresolved entities to LLM
        let mut llm_link_result = mentedb::EntityLinkResult::default();
        if let Some(cognitive_llm) = &self.cognitive_llm {
            let all_entities_with_context = self.db.entity_names_with_context();
            let all_names: Vec<String> =
                all_entities_with_context.iter().map(|(n, _)| n.clone()).collect();
            let unresolved = self.db.unresolved_entity_names();

            if !unresolved.is_empty() {
                tracing::info!(
                    total_entities = all_names.len(),
                    unresolved = unresolved.len(),
                    "running LLM entity resolution"
                );

                // Build EntityCandidate list with context for ALL entities
                // (LLM needs full picture to group correctly)
                let candidates: Vec<mentedb_cognitive::EntityCandidate> = all_entities_with_context
                    .iter()
                    .map(|(name, ctx)| mentedb_cognitive::EntityCandidate {
                        name: name.clone(),
                        context: ctx.clone(),
                        memory_id: None,
                    })
                    .collect();

                // Batch into groups of 50 to avoid context limit issues
                let batch_size = 50;
                let mut all_merge_groups = Vec::new();

                for chunk in candidates.chunks(batch_size) {
                    match cognitive_llm.resolve_entities(chunk).await {
                        Ok(groups) => {
                            all_merge_groups.extend(groups);
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "enrichment: LLM entity resolution failed");
                        }
                    }
                }

                if !all_merge_groups.is_empty() {
                    // Convert LLM merge groups to engine resolution format
                    let resolutions: Vec<mentedb::EntityLinkResolution> = all_merge_groups
                        .iter()
                        .map(|g| mentedb::EntityLinkResolution {
                            canonical: g.canonical.clone(),
                            aliases: g.aliases.clone(),
                            confidence: g.confidence,
                        })
                        .collect();

                    // Identify entities NOT in any merge group → confirmed singletons.
                    // If two unresolved entities are both absent from merge groups,
                    // the LLM implicitly said they're different.
                    let mut grouped_names: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for g in &all_merge_groups {
                        grouped_names.insert(g.canonical.to_lowercase());
                        for a in &g.aliases {
                            grouped_names.insert(a.to_lowercase());
                        }
                    }
                    // We don't negative-cache singletons against each other
                    // because they might just not have been seen together yet.
                    let separations: Vec<mentedb::EntitySeparation> = Vec::new();

                    match self
                        .db
                        .apply_entity_link_resolutions(&resolutions, &separations)
                    {
                        Ok(r) => {
                            llm_link_result = r;
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "enrichment: failed to apply entity resolutions"
                            );
                        }
                    }
                }
            }
        }
        total_edges += llm_link_result.edges_created;

        self.db.mark_enrichment_complete(current_turn);

        tracing::info!(
            stored = total_stored,
            edges = total_edges,
            entities = total_entities,
            sync_linked = sync_link_result.linked,
            llm_linked = llm_link_result.linked,
            llm_edges = llm_link_result.edges_created,
            duplicates_skipped = total_dupes,
            contradictions = total_contradictions,
            batches = batches.len(),
            "sleeptime enrichment complete"
        );
    }

    fn build_extraction_config(
        &self,
        provider_name: &str,
        api_key: Option<&str>,
    ) -> Option<ExtractionConfig> {
        match provider_name.to_lowercase().as_str() {
            "openai" => {
                let key = api_key?;
                let mut cfg = ExtractionConfig::openai(key);
                if let Some(model) = &self.config.llm_model {
                    cfg.model = model.clone();
                }
                Some(cfg)
            }
            "anthropic" => {
                let key = api_key?;
                let mut cfg = ExtractionConfig::anthropic(key);
                if let Some(model) = &self.config.llm_model {
                    cfg.model = model.clone();
                }
                Some(cfg)
            }
            "ollama" => {
                let mut cfg = ExtractionConfig::ollama();
                if let Some(model) = &self.config.llm_model {
                    cfg.model = model.clone();
                }
                Some(cfg)
            }
            _ => None,
        }
    }
}

/// Split memories into batches of at most `batch_size`.
fn batch_conversations(memories: &[MemoryNode], batch_size: usize) -> Vec<Vec<MemoryNode>> {
    memories
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}
