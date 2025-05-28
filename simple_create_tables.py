from connect_db import Base, engine
from models.models import User, Preferences, Reminder, Note, LedgerEntry, Friendship, Permission, Embedding, HistoryLog

print("Creating database tables...")
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")
except Exception as e:
    print(f"❌ Error: {e}") 