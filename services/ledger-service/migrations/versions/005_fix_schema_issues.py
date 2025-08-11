"""Fix schema issues and ensure all required columns exist

Revision ID: 005_fix_schema_issues
Revises: 003_add_friendships_table
Create Date: 2025-01-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '005_fix_schema_issues'
down_revision = '004'
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

def table_exists(table_name):
    """Check if a table exists"""
    connection = op.get_bind()
    result = connection.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name = :table_name"
    ), {"table_name": table_name})
    return result.fetchone() is not None

def upgrade() -> None:
    """Ensure all required columns and tables exist"""
    
    # Ensure ledger_entries table exists
    if not table_exists('ledger_entries'):
        print("Creating ledger_entries table...")
        op.create_table('ledger_entries',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('customer_id', sa.Integer(), nullable=False),
            sa.Column('friend_id', sa.Integer(), nullable=True),
            sa.Column('amount', sa.Numeric(precision=10, scale=2), nullable=False),
            sa.Column('ledger_direction_id', sa.Integer(), nullable=False),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('status', sa.String(length=20), nullable=False, server_default='saved'),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
            sa.Column('friend_name', sa.String(length=255), nullable=True),
            sa.Column('friend_phone', sa.String(length=20), nullable=True),
            sa.Column('friend_email', sa.String(length=255), nullable=True),
            sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ),
            sa.ForeignKeyConstraint(['friend_id'], ['customers.id'], ),
            sa.ForeignKeyConstraint(['ledger_direction_id'], ['ledger_direction.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_ledger_entries_id'), 'ledger_entries', ['id'], unique=False)
    else:
        print("ledger_entries table exists, checking columns...")
        
        # Check and add missing columns
        if not column_exists('ledger_entries', 'status'):
            print("Adding status column...")
            op.add_column('ledger_entries', sa.Column('status', sa.String(length=20), nullable=False, server_default='saved'))
        
        if not column_exists('ledger_entries', 'friend_name'):
            print("Adding friend_name column...")
            op.add_column('ledger_entries', sa.Column('friend_name', sa.String(length=255), nullable=True))
        
        if not column_exists('ledger_entries', 'friend_phone'):
            print("Adding friend_phone column...")
            op.add_column('ledger_entries', sa.Column('friend_phone', sa.String(length=20), nullable=True))
        
        if not column_exists('ledger_entries', 'friend_email'):
            print("Adding friend_email column...")
            op.add_column('ledger_entries', sa.Column('friend_email', sa.String(length=255), nullable=True))
        
        # Ensure friend_id is nullable
        try:
            op.alter_column('ledger_entries', 'friend_id',
                           existing_type=sa.INTEGER(),
                           nullable=True)
            print("Made friend_id nullable")
        except Exception as e:
            print(f"friend_id column modification skipped: {e}")
        
        # Ensure amount is not nullable
        try:
            op.alter_column('ledger_entries', 'amount',
                           existing_type=sa.NUMERIC(precision=10, scale=2),
                           nullable=False)
            print("Made amount not nullable")
        except Exception as e:
            print(f"amount column modification skipped: {e}")
    
    # Ensure ledger_direction table exists
    if not table_exists('ledger_direction'):
        print("Creating ledger_direction table...")
        op.create_table('ledger_direction',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(), nullable=False),
            sa.Column('description', sa.String(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('name')
        )
        op.create_index(op.f('ix_ledger_direction_id'), 'ledger_direction', ['id'], unique=False)
        
        # Insert default direction values
        connection = op.get_bind()
        connection.execute(text(
            "INSERT INTO ledger_direction (id, name, description, is_active) VALUES "
            "(1, 'incoming', 'Money coming in', true), "
            "(2, 'outgoing', 'Money going out', true), "
            "(3, 'settled', 'Debt settled', true)"
        ))
        print("Inserted default ledger directions")
    
    # Ensure customers table exists (basic version)
    if not table_exists('customers'):
        print("Creating basic customers table...")
        op.create_table('customers',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('email', sa.String(), nullable=False),
            sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('email')
        )
        op.create_index(op.f('ix_customers_id'), 'customers', ['id'], unique=False)
        op.create_index(op.f('ix_customers_email'), 'customers', ['email'], unique=False)
        
        # Insert a test customer
        connection = op.get_bind()
        connection.execute(text(
            "INSERT INTO customers (id, email, is_active) VALUES "
            "(1, 'test@example.com', true)"
        ))
        print("Inserted test customer")

def downgrade() -> None:
    """Downgrade operations"""
    # Note: This is a comprehensive fix migration, downgrade would be complex
    # In production, consider the implications before running downgrade
    pass