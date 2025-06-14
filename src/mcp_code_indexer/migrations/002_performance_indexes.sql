-- Performance optimization migration for MCP Code Indexer
-- Adds additional indexes and optimizations for common query patterns

-- Additional composite indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_file_descriptions_project_branch_path ON file_descriptions(project_id, branch, file_path);
CREATE INDEX IF NOT EXISTS idx_file_descriptions_project_modified ON file_descriptions(project_id, last_modified DESC);
CREATE INDEX IF NOT EXISTS idx_file_descriptions_source_project ON file_descriptions(source_project_id) WHERE source_project_id IS NOT NULL;

-- Index for merge conflicts by project and branches
CREATE INDEX IF NOT EXISTS idx_merge_conflicts_project_branches ON merge_conflicts(project_id, source_branch, target_branch, created DESC);

-- Index for token cache expiration cleanup
CREATE INDEX IF NOT EXISTS idx_token_cache_created ON token_cache(created DESC);

-- Optimize FTS5 for prefix searches and ranking
DROP TRIGGER IF EXISTS file_descriptions_ai;
DROP TRIGGER IF EXISTS file_descriptions_ad; 
DROP TRIGGER IF EXISTS file_descriptions_au;
DROP TABLE IF EXISTS file_descriptions_fts;

-- Recreate FTS table with better configuration
CREATE VIRTUAL TABLE file_descriptions_fts USING fts5(
    project_id UNINDEXED,
    branch UNINDEXED,
    file_path,
    description,
    content='file_descriptions',
    content_rowid='rowid',
    prefix='2 3 4'  -- Enable prefix searching for 2-4 character prefixes
);

-- Recreate triggers for FTS sync
CREATE TRIGGER file_descriptions_ai AFTER INSERT ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(rowid, project_id, branch, file_path, description)
  VALUES (new.rowid, new.project_id, new.branch, new.file_path, new.description);
END;

CREATE TRIGGER file_descriptions_ad AFTER DELETE ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(file_descriptions_fts, rowid, project_id, branch, file_path, description)
  VALUES ('delete', old.rowid, old.project_id, old.branch, old.file_path, old.description);
END;

CREATE TRIGGER file_descriptions_au AFTER UPDATE ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(file_descriptions_fts, rowid, project_id, branch, file_path, description)
  VALUES ('delete', old.rowid, old.project_id, old.branch, old.file_path, old.description);
  INSERT INTO file_descriptions_fts(rowid, project_id, branch, file_path, description)
  VALUES (new.rowid, new.project_id, new.branch, new.file_path, new.description);
END;

-- Rebuild FTS index if there's existing data
INSERT INTO file_descriptions_fts(file_descriptions_fts) VALUES('rebuild');

-- Set SQLite optimizations
PRAGMA optimize;
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456; -- 256MB memory mapping
PRAGMA cache_size = -64000;   -- 64MB cache

-- Update statistics for query planner
ANALYZE;
