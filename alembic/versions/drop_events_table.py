"""Drop events table

Revision ID: drop_events_table
Revises: 16faa98b6e03
Create Date: 2025-06-04 13:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'drop_events_table'
down_revision = '16faa98b6e03'
branch_labels = None
depends_on = None

def upgrade():
    # Drop the events table
    op.drop_table('events')

def downgrade():
    # Recreate the events table
    op.create_table('events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('time', sa.TIMESTAMP(), nullable=True),
        sa.Column('attendees', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    ) 