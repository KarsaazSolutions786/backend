"""Ensure friend contact fields exist in ledger entries

Revision ID: 004
Revises: 003
Create Date: 2025-08-11 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003_add_friendships_table'
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
    # Check and add friend_name column if it doesn't exist
    if not column_exists('ledger_entries', 'friend_name'):
        op.add_column('ledger_entries', sa.Column('friend_name', sa.String(length=255), nullable=True))
        print("Added friend_name column")
    
    # Check and add friend_phone column if it doesn't exist
    if not column_exists('ledger_entries', 'friend_phone'):
        op.add_column('ledger_entries', sa.Column('friend_phone', sa.String(length=20), nullable=True))
        print("Added friend_phone column")
    
    # Check and add friend_email column if it doesn't exist
    if not column_exists('ledger_entries', 'friend_email'):
        op.add_column('ledger_entries', sa.Column('friend_email', sa.String(length=255), nullable=True))
        print("Added friend_email column")
    
    # Ensure friend_id is nullable
    try:
        op.alter_column('ledger_entries', 'friend_id',
                        existing_type=sa.INTEGER(),
                        nullable=True)
        print("Made friend_id nullable")
    except Exception as e:
        print(f"friend_id column modification skipped: {e}")
    
    # Ensure amount is NOT NULL
    try:
        op.alter_column('ledger_entries', 'amount',
                        existing_type=sa.NUMERIC(precision=10, scale=2),
                        nullable=False)
        print("Made amount NOT NULL")
    except Exception as e:
        print(f"amount column modification skipped: {e}")

def downgrade() -> None:
    # Remove the friend contact columns if they exist
    if column_exists('ledger_entries', 'friend_email'):
        op.drop_column('ledger_entries', 'friend_email')
    
    if column_exists('ledger_entries', 'friend_phone'):
        op.drop_column('ledger_entries', 'friend_phone')
    
    if column_exists('ledger_entries', 'friend_name'):
        op.drop_column('ledger_entries', 'friend_name')
    
    # Revert friend_id to NOT NULL (if needed)
    try:
        op.alter_column('ledger_entries', 'friend_id',
                        existing_type=sa.INTEGER(),
                        nullable=False)
    except Exception as e:
        print(f"friend_id revert skipped: {e}")
    
    # Revert amount column to nullable (if needed)
    try:
        op.alter_column('ledger_entries', 'amount',
                        existing_type=sa.NUMERIC(precision=10, scale=2),
                        nullable=True)
    except Exception as e:
        print(f"amount revert skipped: {e}")