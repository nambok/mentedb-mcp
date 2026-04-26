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

        // Phase 2: Entity linking — create edges between same-name entities
        let link_result = match self.db.link_entities() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = %e, "enrichment: entity linking failed");
                mentedb::EntityLinkResult::default()
            }
        };
        total_edges += link_result.edges_created;

        self.db.mark_enrichment_complete(current_turn);

        tracing::info!(
            stored = total_stored,
            edges = total_edges,
            entities = total_entities,
            entities_linked = link_result.linked,
            entities_ambiguous = link_result.ambiguous,
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
