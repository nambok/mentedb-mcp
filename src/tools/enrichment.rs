use super::*;

impl MenteDbServer {
    /// Run the sleeptime enrichment pipeline if pending.
    ///
    /// Delegates to the engine's `enrichment::run_enrichment()` orchestrator,
    /// which handles all 4 phases: extraction, entity linking, community
    /// detection, and user model generation.
    pub(super) async fn maybe_run_enrichment(&self, current_turn: u64) {
        if !self.db.needs_enrichment() {
            return;
        }

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

        let result = mentedb::enrichment::run_enrichment(
            &self.db,
            config,
            self.embedding_provider.as_ref(),
            self.cognitive_llm.as_deref(),
            current_turn,
        )
        .await;

        tracing::info!(
            stored = result.memories_stored,
            edges = result.edges_created,
            entities = result.entities_extracted,
            communities = result.communities_created,
            user_model = result.user_model_updated,
            "enrichment delegated to engine"
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
