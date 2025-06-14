-- Migration 005: Remove git remote dependencies from projects table
-- This migration removes remote_origin and upstream_origin columns and their indexes
-- Project identification now relies solely on project name and folder paths

-- Ensure WAL mode is enabled for safe migrations
PRAGMA journal_mode=WAL;

-- Temporarily disable foreign key constraints for migration
PRAGMA foreign_keys=OFF;

-- Start transaction for atomic migration
BEGIN TRANSACTION;

-- Create new projects table without git remote columns
CREATE TABLE projects_new (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT DEFAULT '[]', -- JSON array of aliases
    created DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for the new table (without remote indexes)
CREATE INDEX idx_projects_new_name ON projects_new(name);

-- Migrate data from old table (dropping remote_origin and upstream_origin)
INSERT INTO projects_new (id, name, aliases, created, last_accessed)
SELECT id, name, aliases, created, last_accessed
FROM projects;

-- Drop old table
DROP TABLE projects;

-- Rename new table to original name
ALTER TABLE projects_new RENAME TO projects;

-- Re-enable foreign key constraints
PRAGMA foreign_keys=ON;

-- Commit the migration
COMMIT;
