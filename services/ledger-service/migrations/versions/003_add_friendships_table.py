"""Add friendships table

Revision ID: 003_add_friendships_table
Revises: 002_add_status_field
Create Date: 2024-01-01 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003_add_friendships_table'
down_revision = '002_add_status_field'
branch_labels = None
depends_on = None


def upgrade():
    # Create friendships table
    op.create_table('friendships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=False),
        sa.Column('friend_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('accepted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ),
        sa.ForeignKeyConstraint(['friend_id'], ['customers.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_friendships_id'), 'friendships', ['id'], unique=False)


def downgrade():
    # Drop friendships table
    op.drop_index(op.f('ix_friendships_id'), table_name='friendships')
    op.drop_table('friendships')