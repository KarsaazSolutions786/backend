"""Add status column to ledger entries

Revision ID: 006_add_status_column
Revises: 005_fix_schema_issues
Create Date: 2025-01-20 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '006_add_status_column'
down_revision = '005_fix_schema_issues'
branch_labels = None
depends_on = None

def column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    connection = op.get_bind()
    result = connection.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = :table_name AND column_name = :column_name"
    ), {"table_name": table_name, "column_name": column_name})
    return result.fetchone() is not None

def upgrade() -> None:
    """Add status column if it doesn't exist"""
    
    # Check and add status column if it doesn't exist
    if not column_exists('ledger_entries', 'status'):
        print("Adding status column to ledger_entries...")
        op.add_column('ledger_entries', sa.Column('status', sa.String(length=20), nullable=False, server_default='saved'))
        print("✅ Status column added successfully")
    else:
        print("✅ Status column already exists")

def downgrade() -> None:
    """Remove status column"""
    if column_exists('ledger_entries', 'status'):
        op.drop_column('ledger_entries', 'status')