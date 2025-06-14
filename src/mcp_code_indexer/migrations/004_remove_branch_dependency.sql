-- Migration 004: Remove branch dependency from database schema
-- This migration consolidates multi-branch data and simplifies the schema
-- by removing branch columns from file_descriptions and project_overviews tables

-- Ensure WAL mode is enabled for safe migrations
PRAGMA journal_mode=WAL;

-- Temporarily disable foreign key constraints for migration
PRAGMA foreign_keys=OFF;

-- Start transaction for atomic migration
BEGIN TRANSACTION;

-- Create new file_descriptions table without branch dependency
CREATE TABLE file_descriptions_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    description TEXT NOT NULL,
    file_hash TEXT,
    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    source_project_id TEXT,
    to_be_cleaned INTEGER DEFAULT NULL, -- UNIX timestamp for cleanup, NULL = active
    UNIQUE(project_id, file_path),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (source_project_id) REFERENCES projects(id) ON DELETE SET NULL
);

-- Create indexes for the new table
CREATE INDEX idx_file_descriptions_new_project_id ON file_descriptions_new(project_id);
CREATE INDEX idx_file_descriptions_new_file_hash ON file_descriptions_new(file_hash);
CREATE INDEX idx_file_descriptions_new_last_modified ON file_descriptions_new(last_modified);
CREATE INDEX idx_file_descriptions_new_to_be_cleaned ON file_descriptions_new(to_be_cleaned);

-- Clean up orphaned data before consolidation
-- Remove file_descriptions that reference non-existent projects
DELETE FROM file_descriptions 
WHERE project_id NOT IN (SELECT id FROM projects);

-- Remove file_descriptions with invalid source_project_id
UPDATE file_descriptions 
SET source_project_id = NULL 
WHERE source_project_id IS NOT NULL 
  AND source_project_id NOT IN (SELECT id FROM projects);

-- Consolidate data from old table - keep most recent description per file
-- This handles multi-branch scenarios by selecting the newest data
INSERT INTO file_descriptions_new (
    project_id, file_path, description, file_hash, last_modified, version, source_project_id
)
SELECT 
    project_id,
    file_path,
    description,
    file_hash,
    last_modified,
    version,
    source_project_id
FROM (
    SELECT 
        project_id,
        file_path,
        description,
        file_hash,
        last_modified,
        version,
        source_project_id,
        ROW_NUMBER() OVER (
            PARTITION BY project_id, file_path 
            ORDER BY last_modified DESC
        ) as rn
    FROM file_descriptions
) ranked_descriptions
WHERE rn = 1;

-- Create new project_overviews table without branch dependency
CREATE TABLE project_overviews_new (
    project_id TEXT PRIMARY KEY,
    overview TEXT NOT NULL,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_files INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Create indexes for the new table
CREATE INDEX idx_project_overviews_new_last_modified ON project_overviews_new(last_modified);

-- Clean up orphaned project overviews
DELETE FROM project_overviews 
WHERE project_id NOT IN (SELECT id FROM projects);

-- Consolidate project overviews - keep the one with most tokens (most comprehensive)
INSERT INTO project_overviews_new (
    project_id, overview, last_modified, total_files, total_tokens
)
SELECT 
    project_id,
    overview,
    last_modified,
    total_files,
    total_tokens
FROM (
    SELECT 
        project_id,
        overview,
        last_modified,
        total_files,
        total_tokens,
        ROW_NUMBER() OVER (
            PARTITION BY project_id 
            ORDER BY total_tokens DESC, last_modified DESC
        ) as rn
    FROM project_overviews
) ranked_overviews
WHERE rn = 1;

-- Drop FTS5 triggers for old table
DROP TRIGGER IF EXISTS file_descriptions_ai;
DROP TRIGGER IF EXISTS file_descriptions_ad;
DROP TRIGGER IF EXISTS file_descriptions_au;

-- Drop FTS5 virtual table
DROP TABLE IF EXISTS file_descriptions_fts;

-- Drop old tables
DROP TABLE file_descriptions;
DROP TABLE project_overviews;

-- Rename new tables to original names
ALTER TABLE file_descriptions_new RENAME TO file_descriptions;
ALTER TABLE project_overviews_new RENAME TO project_overviews;

-- Create new FTS5 virtual table without branch column
CREATE VIRTUAL TABLE file_descriptions_fts USING fts5(
    project_id,
    file_path,
    description,
    content='file_descriptions',
    content_rowid='id'
);

-- Populate FTS5 table with existing data (only active records)
INSERT INTO file_descriptions_fts(rowid, project_id, file_path, description)
SELECT id, project_id, file_path, description 
FROM file_descriptions 
WHERE to_be_cleaned IS NULL;

-- Create new FTS5 triggers for the updated schema
CREATE TRIGGER file_descriptions_ai AFTER INSERT ON file_descriptions BEGIN
  -- Only index active records (not marked for cleanup)
  INSERT INTO file_descriptions_fts(rowid, project_id, file_path, description)
  SELECT new.id, new.project_id, new.file_path, new.description
  WHERE new.to_be_cleaned IS NULL;
END;

CREATE TRIGGER file_descriptions_ad AFTER DELETE ON file_descriptions BEGIN
  INSERT INTO file_descriptions_fts(file_descriptions_fts, rowid, project_id, file_path, description)
  VALUES ('delete', old.id, old.project_id, old.file_path, old.description);
END;

CREATE TRIGGER file_descriptions_au AFTER UPDATE ON file_descriptions BEGIN
  -- Remove old record from FTS
  INSERT INTO file_descriptions_fts(file_descriptions_fts, rowid, project_id, file_path, description)
  VALUES ('delete', old.id, old.project_id, old.file_path, old.description);
  
  -- Add new record only if it's active (not marked for cleanup)
  INSERT INTO file_descriptions_fts(rowid, project_id, file_path, description)
  SELECT new.id, new.project_id, new.file_path, new.description
  WHERE new.to_be_cleaned IS NULL;
END;

-- Update merge_conflicts table to remove branch references (optional cleanup)
-- This table structure can remain as-is since it's used for temporary conflict resolution
-- but we'll remove unused indexes that reference branches
DROP INDEX IF EXISTS idx_merge_conflicts_project;
CREATE INDEX idx_merge_conflicts_project ON merge_conflicts(project_id, created);

-- Re-enable foreign key constraints
PRAGMA foreign_keys=ON;

-- Commit the migration
COMMIT;
