// Server construction and lifecycle management.
// The MenteDbServer struct and its MCP handler implementation live in tools.rs.
// This module provides the server startup function.

use std::path::Path;

use mentedb::MenteDb;
use rmcp::ServiceExt;
use rmcp::transport::io::stdio;

use crate::config::ServerConfig;
use crate::tools::MenteDbServer;

/// Start the MCP server on stdio transport.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    tracing::info!("Opening MenteDB at {}", config.data_dir.display());

    std::fs::create_dir_all(&config.data_dir)?;

    let db = MenteDb::open(Path::new(&config.data_dir))?;
    let server = MenteDbServer::new(db, config);

    // Keep a reference to the DB for graceful shutdown
    let db_ref = server.db_ref();

    tracing::info!("Starting MCP server on stdio transport");

    let service = server.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("Server error: {:?}", e);
    })?;

    service.waiting().await?;

    // Graceful shutdown: flush indexes, graph, and WAL to disk
    tracing::info!("Shutting down — flushing database to disk");
    let mut db = db_ref.lock().await;
    if let Err(e) = db.close() {
        tracing::error!(error = %e, "Failed to close database cleanly");
    } else {
        tracing::info!("Database closed cleanly");
    }

    tracing::info!("MCP server shut down");
    Ok(())
}
