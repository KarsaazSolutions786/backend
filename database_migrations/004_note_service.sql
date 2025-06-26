-- Note Service Database Schema
\c note_db;

-- Drop tables if they exist
DROP TABLE IF EXISTS note_shares CASCADE;
DROP TABLE IF EXISTS notes CASCADE;
DROP TABLE IF EXISTS note_folders CASCADE;

-- Note folders
CREATE TABLE note_folders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    color VARCHAR DEFAULT '#1f77b4',
    parent_folder_id UUID REFERENCES note_folders(id),
    is_shared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Notes table
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    folder_id UUID REFERENCES note_folders(id),
    title VARCHAR NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR DEFAULT 'text',
    tags TEXT[],
    category VARCHAR,
    is_shared BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,
    is_pinned BOOLEAN DEFAULT FALSE,
    word_count INTEGER DEFAULT 0,
    character_count INTEGER DEFAULT 0,
    reading_time INTEGER DEFAULT 0,
    attachment_urls TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Note sharing
CREATE TABLE note_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_id UUID NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    owner_user_id VARCHAR NOT NULL,
    shared_with_user_id VARCHAR NOT NULL,
    can_edit BOOLEAN DEFAULT FALSE,
    can_comment BOOLEAN DEFAULT TRUE,
    status VARCHAR DEFAULT 'pending',
    shared_at TIMESTAMP DEFAULT NOW(),
    responded_at TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_notes_user_id ON notes(user_id);
CREATE INDEX idx_notes_folder_id ON notes(folder_id);
CREATE INDEX idx_notes_created_at ON notes(created_at);
CREATE INDEX idx_notes_updated_at ON notes(updated_at);
CREATE INDEX idx_note_folders_user_id ON note_folders(user_id);
CREATE INDEX idx_note_shares_note_id ON note_shares(note_id);
