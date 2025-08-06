"""Add customer timestamps and new friend actions

Revision ID: 002
Revises: 001
Create Date: 2024-12-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # Add created_at and updated_at columns to customers table
    op.add_column('customers', sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True))
    op.add_column('customers', sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True))
    
    # Update existing customers with current timestamp
    connection = op.get_bind()
    connection.execute(text("""
        UPDATE customers 
        SET created_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP 
        WHERE created_at IS NULL OR updated_at IS NULL
    """))
    
    # Make the columns non-nullable after setting default values
    op.alter_column('customers', 'created_at', nullable=False)
    op.alter_column('customers', 'updated_at', nullable=False)

def downgrade():
    # Remove the added columns
    op.drop_column('customers', 'updated_at')
    op.drop_column('customers', 'created_at')