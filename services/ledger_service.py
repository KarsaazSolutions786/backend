from utils.logger import logger

class LedgerService:
    """Service for managing expenses and financial tracking."""
    
    def __init__(self):
        self.expenses_db = {}
    
    async def add_expense(self, user_id: str, amount: float, description: str) -> dict:
        """Add a new expense."""
        expense_id = f"expense_{len(self.expenses_db) + 1}"
        expense = {
            "id": expense_id,
            "user_id": user_id,
            "amount": amount,
            "description": description
        }
        self.expenses_db[expense_id] = expense
        return expense 