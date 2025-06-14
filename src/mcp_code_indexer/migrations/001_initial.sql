-- Initial database schema for MCP Code Indexer
-- Creates tables for projects, file descriptions, and search functionality

-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Enable foreign key support
PRAGMA foreign_keys=ON;

-- Projects table: stores project metadata and identification
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    remote_origin TEXT,
    upstream_origin TEXT,
    aliases TEXT DEFAULT '[]', -- JSON array of aliases
    created DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for project lookups by various identifiers
CREATE INDEX IF NOT EXISTS idx_projects_remote_origin ON projects(remote_origin);
CREATE INDEX IF NOT EXISTS idx_projects_upstream_origin ON projects(upstream_origin);
CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);

-- File descriptions table: stores file content descriptions
CREATE TABLE IF NOT EXISTS file_descriptions (
    project_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    file_path TEXT NOT NULL,
    description TEXT NOT NULL,
    file_hash TEXT,
    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    source_project_id TEXT,
    PRIMARY KEY (project_id, branch, file_path),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (source_project_id) REFERENCES projects(id) ON DELETE SET NULL
);

-- Indexes for file description queries
CREATE INDEX IF NOT EXISTS idx_file_descriptions_project_branch ON file_descriptions(project_id, branch);
CREATE INDEX IF NOT EXISTS idx_file_descriptions_file_hash ON file_descriptions(file_hash);
CREATE INDEX IF NOT EXISTS idx_file_descriptions_last_modified ON file_descriptions(last_modified);

-- Full-text search table for file descriptions
CREATE VIRTUAL TABLE IF NOT EXISTS file_descriptions_fts USING fts5(
    project_id,
    branch,
    file_path,
    description,
    content='file_descriptions',
    content_rowid='rowid'
);

-- Triggers to keep FTS table in sync with file_descriptions
CREATE TRIGGER IF NOT EXISTS file_descriptions_ai AFTER INSERT ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(rowid, project_id, branch, file_path, description)
  VALUES (new.rowid, new.project_id, new.branch, new.file_path, new.description);
END;

CREATE TRIGGER IF NOT EXISTS file_descriptions_ad AFTER DELETE ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(file_descriptions_fts, rowid, project_id, branch, file_path, description)
  VALUES ('delete', old.rowid, old.project_id, old.branch, old.file_path, old.description);
END;

CREATE TRIGGER IF NOT EXISTS file_descriptions_au AFTER UPDATE ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(file_descriptions_fts, rowid, project_id, branch, file_path, description)
  VALUES ('delete', old.rowid, old.project_id, old.branch, old.file_path, old.description);
  INSERT INTO file_descriptions_fts(rowid, project_id, branch, file_path, description)
  VALUES (new.rowid, new.project_id, new.branch, new.file_path, new.description);
END;

-- Merge conflicts table: temporary storage for merge conflicts
CREATE TABLE IF NOT EXISTS merge_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_branch TEXT NOT NULL,
    target_branch TEXT NOT NULL,
    source_description TEXT NOT NULL,
    target_description TEXT NOT NULL,
    resolution TEXT,
    created DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Index for merge conflict lookups
CREATE INDEX IF NOT EXISTS idx_merge_conflicts_project ON merge_conflicts(project_id, source_branch, target_branch);

-- Token cache table: cache token counts for performance
CREATE TABLE IF NOT EXISTS token_cache (
    cache_key TEXT PRIMARY KEY,
    token_count INTEGER NOT NULL,
    created DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires DATETIME
);

-- Index for token cache cleanup
CREATE INDEX IF NOT EXISTS idx_token_cache_expires ON token_cache(expires);
