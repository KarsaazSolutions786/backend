"""Add status field to ledger entries

Revision ID: 002_add_status_field
Revises: 001_add_friend_contact_fields
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_status_field'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Add status column with default value 'saved'
    op.add_column('ledger_entries', sa.Column('status', sa.String(length=20), nullable=False, server_default='saved'))


def downgrade():
    # Remove status column
    op.drop_column('ledger_entries', 'status')