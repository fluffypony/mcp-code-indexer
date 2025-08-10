-- Migration 006: Add vector mode tables and indexes
-- This migration adds support for semantic search capabilities with embeddings
-- Includes code chunks, Merkle tree nodes, and indexing metadata

-- Ensure WAL mode is enabled for safe migrations
PRAGMA journal_mode=WAL;

-- Temporarily disable foreign key constraints for migration
PRAGMA foreign_keys=OFF;

-- Start transaction for atomic migration
BEGIN TRANSACTION;

-- Create code_chunks table for storing semantic code chunks
CREATE TABLE code_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    project_id TEXT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'generic', -- function, class, method, import, etc.
    name TEXT, -- Name of function/class/etc, can be NULL for generic chunks
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_hash TEXT NOT NULL, -- SHA-256 hash of chunk content
    embedding_id TEXT, -- ID in vector database (Turbopuffer)
    redacted BOOLEAN DEFAULT FALSE, -- Whether content was redacted for security
    metadata TEXT DEFAULT '{}', -- JSON metadata about the chunk
    created DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES file_descriptions(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Create indexes for code_chunks table
CREATE INDEX idx_code_chunks_file_id ON code_chunks(file_id);
CREATE INDEX idx_code_chunks_project_id ON code_chunks(project_id);
CREATE INDEX idx_code_chunks_chunk_type ON code_chunks(chunk_type);
CREATE INDEX idx_code_chunks_content_hash ON code_chunks(content_hash);
CREATE INDEX idx_code_chunks_embedding_id ON code_chunks(embedding_id);
CREATE INDEX idx_code_chunks_last_modified ON code_chunks(last_modified);
CREATE INDEX idx_code_chunks_redacted ON code_chunks(redacted);

-- Create merkle_nodes table for efficient change detection
CREATE TABLE merkle_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    path TEXT NOT NULL, -- File/directory path relative to project root
    hash TEXT NOT NULL, -- SHA-256 hash of content or children
    node_type TEXT NOT NULL DEFAULT 'file', -- file, directory, project
    parent_path TEXT, -- Path to parent directory, NULL for root
    children_hash TEXT, -- Combined hash of children for directories
    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, path),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Create indexes for merkle_nodes table
CREATE INDEX idx_merkle_nodes_project_id ON merkle_nodes(project_id);
CREATE INDEX idx_merkle_nodes_path ON merkle_nodes(path);
CREATE INDEX idx_merkle_nodes_hash ON merkle_nodes(hash);
CREATE INDEX idx_merkle_nodes_node_type ON merkle_nodes(node_type);
CREATE INDEX idx_merkle_nodes_parent_path ON merkle_nodes(parent_path);
CREATE INDEX idx_merkle_nodes_last_modified ON merkle_nodes(last_modified);

-- Create index_meta table for tracking vector indexing progress
CREATE TABLE index_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL UNIQUE,
    total_chunks INTEGER DEFAULT 0,
    indexed_chunks INTEGER DEFAULT 0,
    total_files INTEGER DEFAULT 0,
    indexed_files INTEGER DEFAULT 0,
    last_sync DATETIME,
    sync_status TEXT DEFAULT 'pending', -- pending, in_progress, completed, failed, paused
    error_message TEXT,
    queue_depth INTEGER DEFAULT 0,
    processing_rate REAL DEFAULT 0.0, -- Files per second
    estimated_completion DATETIME,
    metadata TEXT DEFAULT '{}', -- JSON metadata
    created DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Create indexes for index_meta table
CREATE INDEX idx_index_meta_project_id ON index_meta(project_id);
CREATE INDEX idx_index_meta_sync_status ON index_meta(sync_status);
CREATE INDEX idx_index_meta_last_sync ON index_meta(last_sync);
CREATE INDEX idx_index_meta_last_modified ON index_meta(last_modified);

-- Add vector_mode column to projects table to track which projects use vector search
ALTER TABLE projects ADD COLUMN vector_mode BOOLEAN DEFAULT FALSE;
CREATE INDEX idx_projects_vector_mode ON projects(vector_mode);

-- Create triggers to maintain consistency between file_descriptions and code_chunks
CREATE TRIGGER code_chunks_cleanup_on_file_delete 
AFTER DELETE ON file_descriptions
BEGIN
    DELETE FROM code_chunks WHERE file_id = OLD.id;
END;

-- Create triggers to update index_meta when chunks are added/removed
CREATE TRIGGER update_index_meta_on_chunk_insert
AFTER INSERT ON code_chunks
BEGIN
    INSERT OR REPLACE INTO index_meta (
        project_id, total_chunks, indexed_chunks, total_files, indexed_files, last_modified
    )
    SELECT 
        NEW.project_id,
        COUNT(*) as total_chunks,
        COUNT(embedding_id) as indexed_chunks,
        (SELECT COUNT(DISTINCT file_id) FROM code_chunks WHERE project_id = NEW.project_id) as total_files,
        (SELECT COUNT(DISTINCT file_id) FROM code_chunks WHERE project_id = NEW.project_id AND embedding_id IS NOT NULL) as indexed_files,
        CURRENT_TIMESTAMP
    FROM code_chunks 
    WHERE project_id = NEW.project_id;
END;

CREATE TRIGGER update_index_meta_on_chunk_update
AFTER UPDATE ON code_chunks
BEGIN
    UPDATE index_meta SET
        indexed_chunks = (
            SELECT COUNT(*) FROM code_chunks 
            WHERE project_id = NEW.project_id AND embedding_id IS NOT NULL
        ),
        indexed_files = (
            SELECT COUNT(DISTINCT file_id) FROM code_chunks 
            WHERE project_id = NEW.project_id AND embedding_id IS NOT NULL
        ),
        last_modified = CURRENT_TIMESTAMP
    WHERE project_id = NEW.project_id;
END;

CREATE TRIGGER update_index_meta_on_chunk_delete
AFTER DELETE ON code_chunks
BEGIN
    UPDATE index_meta SET
        total_chunks = (
            SELECT COUNT(*) FROM code_chunks 
            WHERE project_id = OLD.project_id
        ),
        indexed_chunks = (
            SELECT COUNT(*) FROM code_chunks 
            WHERE project_id = OLD.project_id AND embedding_id IS NOT NULL
        ),
        total_files = (
            SELECT COUNT(DISTINCT file_id) FROM code_chunks 
            WHERE project_id = OLD.project_id
        ),
        indexed_files = (
            SELECT COUNT(DISTINCT file_id) FROM code_chunks 
            WHERE project_id = OLD.project_id AND embedding_id IS NOT NULL
        ),
        last_modified = CURRENT_TIMESTAMP
    WHERE project_id = OLD.project_id;
END;

-- Create view for vector search results with file information
CREATE VIEW vector_search_view AS
SELECT 
    cc.id as chunk_id,
    cc.file_id,
    fd.file_path,
    cc.chunk_type,
    cc.name as chunk_name,
    cc.start_line,
    cc.end_line,
    cc.content_hash,
    cc.embedding_id,
    cc.redacted,
    cc.metadata as chunk_metadata,
    cc.project_id,
    p.name as project_name,
    fd.description as file_description,
    cc.created as chunk_created,
    cc.last_modified as chunk_modified,
    fd.last_modified as file_modified
FROM code_chunks cc
JOIN file_descriptions fd ON cc.file_id = fd.id
JOIN projects p ON cc.project_id = p.id
WHERE cc.embedding_id IS NOT NULL
  AND fd.to_be_cleaned IS NULL;

-- Re-enable foreign key constraints
PRAGMA foreign_keys=ON;

-- Commit the migration
COMMIT;
