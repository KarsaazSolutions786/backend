"""Add friend contact fields to ledger entries

Revision ID: 001
Revises: 
Create Date: 2024-12-19 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Modify the friend_id column to be nullable
    op.alter_column('ledger_entries', 'friend_id',
                    existing_type=sa.INTEGER(),
                    nullable=True)
    
    # Add new columns for non-app contacts
    op.add_column('ledger_entries', sa.Column('friend_name', sa.String(length=255), nullable=True))
    op.add_column('ledger_entries', sa.Column('friend_phone', sa.String(length=20), nullable=True))
    op.add_column('ledger_entries', sa.Column('friend_email', sa.String(length=255), nullable=True))
    
    # Modify amount column to be NOT NULL
    op.alter_column('ledger_entries', 'amount',
                    existing_type=sa.NUMERIC(precision=10, scale=2),
                    nullable=False)

def downgrade() -> None:
    # Remove the new columns
    op.drop_column('ledger_entries', 'friend_email')
    op.drop_column('ledger_entries', 'friend_phone')
    op.drop_column('ledger_entries', 'friend_name')
    
    # Revert friend_id to NOT NULL
    op.alter_column('ledger_entries', 'friend_id',
                    existing_type=sa.INTEGER(),
                    nullable=False)
    
    # Revert amount column to nullable
    op.alter_column('ledger_entries', 'amount',
                    existing_type=sa.NUMERIC(precision=10, scale=2),
                    nullable=True)