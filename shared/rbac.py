"""
Role-Based Access Control (RBAC) for Eindr Microservices

This module provides comprehensive role-based access control functionality
for managing permissions across all microservices.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from datetime import datetime, timedelta
from functools import wraps
from fastapi import HTTPException, status, Depends
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Table, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
import json

logger = logging.getLogger(__name__)

# Base for RBAC models
RBACBase = declarative_base()

class Permission(str, Enum):
    """System permissions enum"""
    
    # User management
    CREATE_USER = "user:create"
    READ_USER = "user:read"
    UPDATE_USER = "user:update"
    DELETE_USER = "user:delete"
    MANAGE_USER_ROLES = "user:manage_roles"
    
    # Customer data
    CREATE_CUSTOMER = "customer:create"
    READ_CUSTOMER = "customer:read"
    UPDATE_CUSTOMER = "customer:update"
    DELETE_CUSTOMER = "customer:delete"
    READ_ALL_CUSTOMERS = "customer:read_all"
    
    # Notes
    CREATE_NOTE = "note:create"
    READ_NOTE = "note:read"
    UPDATE_NOTE = "note:update"
    DELETE_NOTE = "note:delete"
    READ_ALL_NOTES = "note:read_all"
    
    # Reminders
    CREATE_REMINDER = "reminder:create"
    READ_REMINDER = "reminder:read"
    UPDATE_REMINDER = "reminder:update"
    DELETE_REMINDER = "reminder:delete"
    READ_ALL_REMINDERS = "reminder:read_all"
    
    # Ledger entries
    CREATE_LEDGER_ENTRY = "ledger:create"
    READ_LEDGER_ENTRY = "ledger:read"
    UPDATE_LEDGER_ENTRY = "ledger:update"
    DELETE_LEDGER_ENTRY = "ledger:delete"
    READ_ALL_LEDGER_ENTRIES = "ledger:read_all"
    
    # Chat conversations
    CREATE_CONVERSATION = "chat:create"
    READ_CONVERSATION = "chat:read"
    UPDATE_CONVERSATION = "chat:update"
    DELETE_CONVERSATION = "chat:delete"
    READ_ALL_CONVERSATIONS = "chat:read_all"
    
    # Friends
    ADD_FRIEND = "friend:add"
    REMOVE_FRIEND = "friend:remove"
    READ_FRIENDS = "friend:read"
    READ_ALL_FRIENDS = "friend:read_all"
    
    # AI Services
    USE_STT = "ai:stt"
    USE_TTS = "ai:tts"
    USE_INTENT_CLASSIFICATION = "ai:intent"
    USE_CHAT_AI = "ai:chat"
    USE_AI_PIPELINE = "ai:pipeline"
    
    # System administration
    VIEW_LOGS = "system:view_logs"
    MANAGE_SYSTEM = "system:manage"
    BACKUP_DATA = "system:backup"
    RESTORE_DATA = "system:restore"
    VIEW_METRICS = "system:metrics"
    
    # Service management
    MANAGE_SERVICES = "service:manage"
    VIEW_SERVICE_STATUS = "service:status"
    RESTART_SERVICES = "service:restart"

class DefaultRole(str, Enum):
    """Default system roles"""
    
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"
    SERVICE_ACCOUNT = "service_account"

# Association table for many-to-many relationship between roles and permissions
role_permissions = Table(
    'role_permissions',
    RBACBase.metadata,
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('permissions.id'), primary_key=True)
)

# Association table for many-to-many relationship between users and roles
user_roles = Table(
    'user_roles',
    RBACBase.metadata,
    Column('customer_id', Integer, ForeignKey('rbac_users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)

class RBACPermission(RBACBase):
    """Permission model for RBAC"""
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    resource = Column(String(50))  # e.g., 'user', 'note', 'system'
    action = Column(String(50))    # e.g., 'create', 'read', 'update', 'delete'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    roles = relationship("RBACRole", secondary=role_permissions, back_populates="permissions")

class RBACRole(RBACBase):
    """Role model for RBAC"""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)
    is_system_role = Column(Boolean, default=False)  # System roles cannot be deleted
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    permissions = relationship("RBACPermission", secondary=role_permissions, back_populates="roles")
    users = relationship("RBACUser", secondary=user_roles, back_populates="roles")

class RBACUser(RBACBase):
    """User model for RBAC (extends customer data)"""
    __tablename__ = "rbac_users"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, unique=True, nullable=False, index=True)  # References customer.id
    email = Column(String(255), unique=True, nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    last_permission_check = Column(DateTime)
    permission_cache = Column(Text)  # JSON string of cached permissions
    cache_expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    roles = relationship("RBACRole", secondary=user_roles, back_populates="users")

class PermissionService:
    """Service for managing permissions and roles"""
    
    def __init__(self, db: Session):
        self.db = db
        self.cache_duration = timedelta(hours=1)  # Cache permissions for 1 hour
    
    def get_user_permissions(self, customer_id: int, use_cache: bool = True) -> Set[str]:
        """
        Get all permissions for a user
        
        Args:
            customer_id: Customer ID
            use_cache: Whether to use cached permissions
            
        Returns:
            Set of permission names
        """
        user = self.db.query(RBACUser).filter(RBACUser.customer_id == customer_id).first()
        if not user:
            return set()
        
        if not user.is_active:
            return set()
        
        # Check cache
        if use_cache and user.permission_cache and user.cache_expires_at:
            if datetime.utcnow() < user.cache_expires_at:
                try:
                    return set(json.loads(user.permission_cache))
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # If superuser, return all permissions
        if user.is_superuser:
            all_permissions = self.db.query(RBACPermission).all()
            permissions = {perm.name for perm in all_permissions}
        else:
            # Get permissions from roles
            permissions = set()
            for role in user.roles:
                if role.is_active:
                    for permission in role.permissions:
                        permissions.add(permission.name)
        
        # Cache the permissions
        user.permission_cache = json.dumps(list(permissions))
        user.cache_expires_at = datetime.utcnow() + self.cache_duration
        user.last_permission_check = datetime.utcnow()
        self.db.commit()
        
        return permissions
    
    def has_permission(self, customer_id: int, permission: Union[str, Permission]) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            customer_id: Customer ID
            permission: Permission name or enum
            
        Returns:
            True if user has permission
        """
        if isinstance(permission, Permission):
            permission = permission.value
        
        permissions = self.get_user_permissions(customer_id)
        return permission in permissions
    
    def has_any_permission(self, customer_id: int, permissions: List[Union[str, Permission]]) -> bool:
        """
        Check if user has any of the specified permissions
        
        Args:
            customer_id: Customer ID
            permissions: List of permission names or enums
            
        Returns:
            True if user has at least one permission
        """
        user_permissions = self.get_user_permissions(customer_id)
        
        for perm in permissions:
            if isinstance(perm, Permission):
                perm = perm.value
            if perm in user_permissions:
                return True
        
        return False
    
    def has_all_permissions(self, customer_id: int, permissions: List[Union[str, Permission]]) -> bool:
        """
        Check if user has all specified permissions
        
        Args:
            customer_id: Customer ID
            permissions: List of permission names or enums
            
        Returns:
            True if user has all permissions
        """
        user_permissions = self.get_user_permissions(customer_id)
        
        for perm in permissions:
            if isinstance(perm, Permission):
                perm = perm.value
            if perm not in user_permissions:
                return False
        
        return True
    
    def assign_role(self, customer_id: int, role_name: str) -> bool:
        """
        Assign a role to a user
        
        Args:
            customer_id: Customer ID
            role_name: Name of role to assign
            
        Returns:
            True if successful
        """
        user = self.db.query(RBACUser).filter(RBACUser.customer_id == customer_id).first()
        if not user:
            return False
        
        role = self.db.query(RBACRole).filter(RBACRole.name == role_name).first()
        if not role or not role.is_active:
            return False
        
        if role not in user.roles:
            user.roles.append(role)
            # Clear permission cache
            user.permission_cache = None
            user.cache_expires_at = None
            self.db.commit()
        
        return True
    
    def remove_role(self, customer_id: int, role_name: str) -> bool:
        """
        Remove a role from a user
        
        Args:
            customer_id: Customer ID
            role_name: Name of role to remove
            
        Returns:
            True if successful
        """
        user = self.db.query(RBACUser).filter(RBACUser.customer_id == customer_id).first()
        if not user:
            return False
        
        role = self.db.query(RBACRole).filter(RBACRole.name == role_name).first()
        if not role:
            return False
        
        if role in user.roles:
            user.roles.remove(role)
            # Clear permission cache
            user.permission_cache = None
            user.cache_expires_at = None
            self.db.commit()
        
        return True
    
    def get_user_roles(self, customer_id: int) -> List[str]:
        """
        Get all roles for a user
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of role names
        """
        user = self.db.query(RBACUser).filter(RBACUser.customer_id == customer_id).first()
        if not user:
            return []
        
        return [role.name for role in user.roles if role.is_active]
    
    def clear_permission_cache(self, customer_id: int = None):
        """
        Clear permission cache for user(s)
        
        Args:
            customer_id: Specific customer ID, or None to clear all caches
        """
        if customer_id:
            user = self.db.query(RBACUser).filter(RBACUser.customer_id == customer_id).first()
            if user:
                user.permission_cache = None
                user.cache_expires_at = None
        else:
            self.db.query(RBACUser).update({
                RBACUser.permission_cache: None,
                RBACUser.cache_expires_at: None
            })
        
        self.db.commit()

class RoleService:
    """Service for managing roles"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_role(self, name: str, description: str = None, permissions: List[str] = None) -> Optional[RBACRole]:
        """
        Create a new role
        
        Args:
            name: Role name
            description: Role description
            permissions: List of permission names to assign
            
        Returns:
            Created role or None if failed
        """
        # Check if role already exists
        existing_role = self.db.query(RBACRole).filter(RBACRole.name == name).first()
        if existing_role:
            return None
        
        role = RBACRole(name=name, description=description)
        self.db.add(role)
        self.db.flush()  # Get the ID
        
        # Assign permissions
        if permissions:
            for perm_name in permissions:
                permission = self.db.query(RBACPermission).filter(RBACPermission.name == perm_name).first()
                if permission:
                    role.permissions.append(permission)
        
        self.db.commit()
        return role
    
    def delete_role(self, role_name: str) -> bool:
        """
        Delete a role (if not system role)
        
        Args:
            role_name: Name of role to delete
            
        Returns:
            True if successful
        """
        role = self.db.query(RBACRole).filter(RBACRole.name == role_name).first()
        if not role or role.is_system_role:
            return False
        
        # Remove role from all users
        for user in role.users:
            user.roles.remove(role)
            # Clear permission cache
            user.permission_cache = None
            user.cache_expires_at = None
        
        self.db.delete(role)
        self.db.commit()
        return True
    
    def add_permission_to_role(self, role_name: str, permission_name: str) -> bool:
        """
        Add permission to role
        
        Args:
            role_name: Role name
            permission_name: Permission name
            
        Returns:
            True if successful
        """
        role = self.db.query(RBACRole).filter(RBACRole.name == role_name).first()
        permission = self.db.query(RBACPermission).filter(RBACPermission.name == permission_name).first()
        
        if not role or not permission:
            return False
        
        if permission not in role.permissions:
            role.permissions.append(permission)
            # Clear permission cache for all users with this role
            for user in role.users:
                user.permission_cache = None
                user.cache_expires_at = None
            self.db.commit()
        
        return True
    
    def remove_permission_from_role(self, role_name: str, permission_name: str) -> bool:
        """
        Remove permission from role
        
        Args:
            role_name: Role name
            permission_name: Permission name
            
        Returns:
            True if successful
        """
        role = self.db.query(RBACRole).filter(RBACRole.name == role_name).first()
        permission = self.db.query(RBACPermission).filter(RBACPermission.name == permission_name).first()
        
        if not role or not permission:
            return False
        
        if permission in role.permissions:
            role.permissions.remove(permission)
            # Clear permission cache for all users with this role
            for user in role.users:
                user.permission_cache = None
                user.cache_expires_at = None
            self.db.commit()
        
        return True

def require_permission(permission: Union[str, Permission]):
    """
    Decorator to require specific permission
    
    Args:
        permission: Required permission
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current customer from kwargs
            current_customer = kwargs.get('current_customer')
            if not current_customer:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            customer_id = current_customer.get('customer_id')
            if not customer_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid user context"
                )
            
            # Get database session
            db = kwargs.get('db')
            if not db:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session not available"
                )
            
            # Check permission
            permission_service = PermissionService(db)
            if not permission_service.has_permission(customer_id, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_any_permission(permissions: List[Union[str, Permission]]):
    """
    Decorator to require any of the specified permissions
    
    Args:
        permissions: List of permissions (user needs at least one)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_customer = kwargs.get('current_customer')
            if not current_customer:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            customer_id = current_customer.get('customer_id')
            db = kwargs.get('db')
            
            permission_service = PermissionService(db)
            if not permission_service.has_any_permission(customer_id, permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role: Union[str, DefaultRole]):
    """
    Decorator to require specific role
    
    Args:
        role: Required role
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_customer = kwargs.get('current_customer')
            if not current_customer:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            customer_id = current_customer.get('customer_id')
            db = kwargs.get('db')
            
            permission_service = PermissionService(db)
            user_roles = permission_service.get_user_roles(customer_id)
            
            role_name = role.value if isinstance(role, DefaultRole) else role
            if role_name not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient role privileges"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def initialize_default_permissions(db: Session):
    """Initialize default permissions in database"""
    default_permissions = [
        # User management
        ("user:create", "Create new users", "user", "create"),
        ("user:read", "Read user information", "user", "read"),
        ("user:update", "Update user information", "user", "update"),
        ("user:delete", "Delete users", "user", "delete"),
        ("user:manage_roles", "Manage user roles", "user", "manage_roles"),
        
        # Customer data
        ("customer:create", "Create customer records", "customer", "create"),
        ("customer:read", "Read customer information", "customer", "read"),
        ("customer:update", "Update customer information", "customer", "update"),
        ("customer:delete", "Delete customer records", "customer", "delete"),
        ("customer:read_all", "Read all customer records", "customer", "read_all"),
        
        # Notes
        ("note:create", "Create notes", "note", "create"),
        ("note:read", "Read notes", "note", "read"),
        ("note:update", "Update notes", "note", "update"),
        ("note:delete", "Delete notes", "note", "delete"),
        ("note:read_all", "Read all notes", "note", "read_all"),
        
        # AI Services
        ("ai:stt", "Use speech-to-text service", "ai", "stt"),
        ("ai:tts", "Use text-to-speech service", "ai", "tts"),
        ("ai:intent", "Use intent classification", "ai", "intent"),
        ("ai:chat", "Use chat AI service", "ai", "chat"),
        ("ai:pipeline", "Use AI pipeline", "ai", "pipeline"),
        
        # System
        ("system:view_logs", "View system logs", "system", "view_logs"),
        ("system:manage", "Manage system", "system", "manage"),
        ("system:backup", "Backup data", "system", "backup"),
        ("system:restore", "Restore data", "system", "restore"),
        ("system:metrics", "View metrics", "system", "metrics"),
    ]
    
    for name, description, resource, action in default_permissions:
        existing = db.query(RBACPermission).filter(RBACPermission.name == name).first()
        if not existing:
            permission = RBACPermission(
                name=name,
                description=description,
                resource=resource,
                action=action
            )
            db.add(permission)
    
    db.commit()

def initialize_default_roles(db: Session):
    """Initialize default roles in database"""
    role_service = RoleService(db)
    
    # Super Admin - all permissions
    all_permissions = [perm.value for perm in Permission]
    role_service.create_role(
        DefaultRole.SUPER_ADMIN.value,
        "Super administrator with all permissions",
        all_permissions
    )
    
    # Admin - most permissions except super admin functions
    admin_permissions = [
        Permission.CREATE_USER.value, Permission.READ_USER.value, Permission.UPDATE_USER.value,
        Permission.CREATE_CUSTOMER.value, Permission.READ_CUSTOMER.value, Permission.UPDATE_CUSTOMER.value,
        Permission.READ_ALL_CUSTOMERS.value, Permission.READ_ALL_NOTES.value,
        Permission.VIEW_LOGS.value, Permission.VIEW_METRICS.value
    ]
    role_service.create_role(
        DefaultRole.ADMIN.value,
        "Administrator with management permissions",
        admin_permissions
    )
    
    # User - basic permissions
    user_permissions = [
        Permission.READ_USER.value, Permission.UPDATE_USER.value,
        Permission.CREATE_NOTE.value, Permission.READ_NOTE.value,
        Permission.UPDATE_NOTE.value, Permission.DELETE_NOTE.value,
        Permission.CREATE_REMINDER.value, Permission.READ_REMINDER.value,
        Permission.UPDATE_REMINDER.value, Permission.DELETE_REMINDER.value,
        Permission.USE_STT.value, Permission.USE_TTS.value,
        Permission.USE_INTENT_CLASSIFICATION.value, Permission.USE_CHAT_AI.value
    ]
    role_service.create_role(
        DefaultRole.USER.value,
        "Regular user with basic permissions",
        user_permissions
    )
    
    # Guest - minimal permissions
    guest_permissions = [
        Permission.READ_USER.value,
        Permission.USE_CHAT_AI.value
    ]
    role_service.create_role(
        DefaultRole.GUEST.value,
        "Guest user with minimal permissions",
        guest_permissions
    )

def setup_rbac_for_customer(db: Session, customer_id: int, email: str, default_role: str = DefaultRole.USER.value):
    """
    Set up RBAC for a new customer
    
    Args:
        db: Database session
        customer_id: Customer ID
        email: Customer email
        default_role: Default role to assign
    """
    # Check if RBAC user already exists
    existing_user = db.query(RBACUser).filter(RBACUser.customer_id == customer_id).first()
    if existing_user:
        return existing_user
    
    # Create RBAC user
    rbac_user = RBACUser(
        customer_id=customer_id,
        email=email
    )
    db.add(rbac_user)
    db.flush()
    
    # Assign default role
    permission_service = PermissionService(db)
    permission_service.assign_role(customer_id, default_role)
    
    db.commit()
    return rbac_user 