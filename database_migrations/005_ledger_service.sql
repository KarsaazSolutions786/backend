-- Ledger Service Database Schema
\c ledger_db;

-- Drop tables if they exist
DROP TABLE IF EXISTS budgets CASCADE;
DROP TABLE IF EXISTS expenses CASCADE;
DROP TABLE IF EXISTS expense_categories CASCADE;

-- Expense categories
CREATE TABLE expense_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    color VARCHAR DEFAULT '#1f77b4',
    icon VARCHAR,
    is_default BOOLEAN DEFAULT FALSE,
    budget_limit DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Expenses table
CREATE TABLE expenses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    category_id UUID REFERENCES expense_categories(id),
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    description TEXT NOT NULL,
    date DATE NOT NULL,
    payment_method VARCHAR,
    payment_account VARCHAR,
    tags TEXT[],
    location_name VARCHAR,
    receipt_urls TEXT[],
    is_recurring BOOLEAN DEFAULT FALSE,
    recurring_frequency VARCHAR,
    recurring_end_date DATE,
    notes TEXT,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    tax_rate DECIMAL(5,2) DEFAULT 0,
    is_business_expense BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Budgets
CREATE TABLE budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    category_id UUID REFERENCES expense_categories(id),
    name VARCHAR NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    period VARCHAR NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    alert_threshold DECIMAL(5,2) DEFAULT 80.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_expenses_user_id ON expenses(user_id);
CREATE INDEX idx_expenses_date ON expenses(date);
CREATE INDEX idx_expenses_category ON expenses(category_id);
CREATE INDEX idx_expenses_amount ON expenses(amount);
CREATE INDEX idx_budgets_user_id ON budgets(user_id);
CREATE INDEX idx_budgets_period ON budgets(start_date, end_date);

-- Insert default categories
INSERT INTO expense_categories (user_id, name, description, color, is_default) VALUES 
('default', 'Food & Dining', 'Restaurants, groceries, and food delivery', '#FF6B6B', TRUE),
('default', 'Transportation', 'Gas, public transit, rideshare, parking', '#4ECDC4', TRUE),
('default', 'Shopping', 'Clothing, electronics, household items', '#45B7D1', TRUE),
('default', 'Entertainment', 'Movies, games, subscriptions, events', '#96CEB4', TRUE),
('default', 'Bills & Utilities', 'Rent, electricity, water, internet', '#FFEAA7', TRUE),
('default', 'Healthcare', 'Medical, dental, pharmacy, insurance', '#DDA0DD', TRUE),
('default', 'Education', 'Books, courses, tuition, supplies', '#98D8C8', TRUE),
('default', 'Travel', 'Flights, hotels, vacation expenses', '#F7DC6F', TRUE),
('default', 'Other', 'Miscellaneous expenses', '#BDC3C7', TRUE);
