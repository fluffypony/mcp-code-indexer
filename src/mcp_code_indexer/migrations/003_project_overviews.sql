-- Migration 003: Add project_overviews table for condensed codebase overviews
-- This table stores comprehensive narrative overviews of entire codebases
-- as an alternative to file-by-file descriptions

CREATE TABLE IF NOT EXISTS project_overviews (
    project_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    overview TEXT NOT NULL,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_files INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, branch),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
) WITHOUT ROWID;

-- Index for temporal queries and overview management
CREATE INDEX IF NOT EXISTS idx_project_overviews_last_modified ON project_overviews(last_modified);

-- Index for project-based queries
CREATE INDEX IF NOT EXISTS idx_project_overviews_project_id ON project_overviews(project_id);
