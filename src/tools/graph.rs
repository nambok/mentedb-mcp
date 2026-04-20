use super::*;

#[rmcp::tool_router(router = tool_router_graph, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(description = "Create a typed relationship edge between two memories.")]
    async fn relate_memories(
        &self,
        Parameters(req): Parameters<RelateMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let from_id = match parse_uuid(&req.from_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let to_id = match parse_uuid(&req.to_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let edge_type = match parse_edge_type(&req.edge_type) {
            Ok(et) => et,
            Err(e) => return error_result(&e),
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let edge = MemoryEdge {
            source: MemoryId(from_id),
            target: MemoryId(to_id),
            edge_type,
            weight: req.weight.unwrap_or(1.0),
            created_at: now,
            valid_from: None,
            valid_until: None,
            label: None,
        };

        let db = &*self.db;
        match db.relate(edge) {
            Ok(()) => {
                tracing::info!(from = %from_id, to = %to_id, edge_type = %req.edge_type, "memories related");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "status": "related",
                        "from": from_id.to_string(),
                        "to": to_id.to_string(),
                        "edge_type": req.edge_type,
                    })
                    .to_string(),
                )]))
            }
            Err(e) => {
                tracing::error!(from = %from_id, to = %to_id, error = %e, "relate_memories failed");
                error_result(&format!("Failed to relate memories: {e}"))
            }
        }
    }

    #[rmcp::tool(
        description = "Find all memories directly related to a given memory, with optional edge type filter and traversal depth."
    )]
    async fn get_related(
        &self,
        Parameters(req): Parameters<GetRelatedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let depth = req.depth.unwrap_or(1);
        let edge_filter: Option<EdgeType> = match req.edge_type.as_deref() {
            Some(et) => match parse_edge_type(et) {
                Ok(et) => Some(et),
                Err(e) => return error_result(&e),
            },
            None => None,
        };

        let db = &*self.db;
        let graph = db.graph();
        let csr = graph.graph();
        let mem_id = MemoryId(id);

        if !csr.contains_node(mem_id) {
            return error_result(&format!("Memory not found in graph: {id}"));
        }

        let mut related: Vec<serde_json::Value> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![(mem_id, 0usize)];
        visited.insert(mem_id);

        while let Some((current, current_depth)) = frontier.pop() {
            if current_depth >= depth {
                continue;
            }
            for (target, edge) in csr.outgoing(current) {
                if let Some(ref filter) = edge_filter
                    && edge.edge_type != *filter
                {
                    continue;
                }
                let next_depth = current_depth + 1;
                if visited.insert(target) {
                    related.push(json!({
                        "id": target.to_string(),
                        "edge_type": format!("{:?}", edge.edge_type),
                        "weight": edge.weight,
                        "depth": next_depth,
                        "direction": "outgoing",
                    }));
                    frontier.push((target, next_depth));
                }
            }
            for (source, edge) in csr.incoming(current) {
                if let Some(ref filter) = edge_filter
                    && edge.edge_type != *filter
                {
                    continue;
                }
                let next_depth = current_depth + 1;
                if visited.insert(source) {
                    related.push(json!({
                        "id": source.to_string(),
                        "edge_type": format!("{:?}", edge.edge_type),
                        "weight": edge.weight,
                        "depth": next_depth,
                        "direction": "incoming",
                    }));
                    frontier.push((source, next_depth));
                }
            }
        }

        tracing::info!(id = %id, depth = depth, related_count = related.len(), "get_related completed");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "id": id.to_string(), "related": related, "count": related.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Find the shortest path between two memories in the knowledge graph."
    )]
    async fn find_path(
        &self,
        Parameters(req): Parameters<FindPathRequest>,
    ) -> Result<CallToolResult, McpError> {
        let from_id = match parse_uuid(&req.from_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let to_id = match parse_uuid(&req.to_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        let csr = db.graph().graph();
        let from_mem = MemoryId(from_id);
        let to_mem = MemoryId(to_id);

        if !csr.contains_node(from_mem) {
            return error_result(&format!("Source memory not found in graph: {from_id}"));
        }
        if !csr.contains_node(to_mem) {
            return error_result(&format!("Target memory not found in graph: {to_id}"));
        }

        match shortest_path(&*csr, from_mem, to_mem) {
            Some(path) => {
                let path_strs: Vec<String> = path.iter().map(|id| id.to_string()).collect();
                tracing::info!(from = %from_id, to = %to_id, hops = path.len() - 1, "path found");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "from": from_id.to_string(),
                        "to": to_id.to_string(),
                        "path": path_strs,
                        "hops": path.len() - 1,
                    })
                    .to_string(),
                )]))
            }
            None => {
                tracing::info!(from = %from_id, to = %to_id, "no path found");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "from": from_id.to_string(),
                        "to": to_id.to_string(),
                        "path": null,
                        "message": "no path found",
                    })
                    .to_string(),
                )]))
            }
        }
    }

    #[rmcp::tool(
        description = "Extract all nodes and edges within N hops of a center memory, returning the local subgraph."
    )]
    async fn get_subgraph(
        &self,
        Parameters(req): Parameters<GetSubgraphRequest>,
    ) -> Result<CallToolResult, McpError> {
        let center_id = match parse_uuid(&req.center_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let radius = req.radius.unwrap_or(2);

        let db = &*self.db;
        let csr = db.graph().graph();
        let center_mem = MemoryId(center_id);

        if !csr.contains_node(center_mem) {
            return error_result(&format!("Center memory not found in graph: {center_id}"));
        }

        let (nodes, edges) = extract_subgraph(&*csr, center_mem, radius);

        let nodes_json: Vec<String> = nodes.iter().map(|id| id.to_string()).collect();
        let edges_json: Vec<serde_json::Value> = edges
            .iter()
            .map(|e| {
                json!({
                    "source": e.source.to_string(),
                    "target": e.target.to_string(),
                    "edge_type": format!("{:?}", e.edge_type),
                    "weight": e.weight,
                })
            })
            .collect();

        tracing::info!(
            center = %center_id,
            radius = radius,
            nodes = nodes.len(),
            edges = edges.len(),
            "subgraph extracted"
        );
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "center": center_id.to_string(),
                "radius": radius,
                "nodes": nodes_json,
                "edges": edges_json,
                "node_count": nodes.len(),
                "edge_count": edges.len(),
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Find all memories that contradict a given memory via Contradicts edges in the knowledge graph."
    )]
    async fn find_contradictions(
        &self,
        Parameters(req): Parameters<FindContradictionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        let csr = db.graph().graph();
        let mem_id = MemoryId(id);

        if !csr.contains_node(mem_id) {
            return error_result(&format!("Memory not found in graph: {id}"));
        }

        let contradictions = find_contradictions(&*csr, mem_id);
        let ids: Vec<String> = contradictions.iter().map(|c| c.to_string()).collect();

        tracing::info!(id = %id, contradictions = contradictions.len(), "contradictions found");
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "id": id.to_string(),
                "contradictions": ids,
                "count": contradictions.len(),
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Propagate a confidence change through the knowledge graph. Returns all affected memories and their new confidence values."
    )]
    async fn propagate_belief(
        &self,
        Parameters(req): Parameters<PropagateBeliefRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        if req.new_confidence < 0.0 || req.new_confidence > 1.0 {
            return error_result("new_confidence must be between 0.0 and 1.0");
        }

        let db = &*self.db;
        let graph = db.graph();
        let mem_id = MemoryId(id);

        if !graph.graph().contains_node(mem_id) {
            return error_result(&format!("Memory not found in graph: {id}"));
        }

        let affected = graph.propagate_belief_change(mem_id, req.new_confidence);
        let affected_json: Vec<serde_json::Value> = affected
            .iter()
            .map(|(mid, conf)| {
                json!({
                    "id": mid.to_string(),
                    "new_confidence": conf,
                })
            })
            .collect();

        // Persist updated confidence values
        let db = &*self.db;
        for (mid, conf) in &affected {
            if let Ok(Some(sm)) = find_memory_by_id(&db, mid.0) {
                let mut updated = sm.memory.clone();
                updated.confidence = *conf;
                let _ = db.store(updated);
            }
        }

        tracing::info!(
            id = %id,
            new_confidence = req.new_confidence,
            affected = affected.len(),
            "belief propagation completed"
        );
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "id": id.to_string(),
                "new_confidence": req.new_confidence,
                "affected": affected_json,
                "affected_count": affected.len(),
            })
            .to_string(),
        )]))
    }
}
