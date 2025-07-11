"""
Database Security for Eindr Microservices

This module provides secure database operations, SQL injection prevention,
and database security utilities for all microservices.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from sqlalchemy import text, create_engine, MetaData, Table, Column
from sqlalchemy.orm import Session, Query
from sqlalchemy.sql import sqltypes
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DataError
import sqlparse
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class SQLInjectionDetector:
    """Detect potential SQL injection attempts"""
    
    # SQL injection patterns to detect
    INJECTION_PATTERNS = [
        # Union-based injection
        r'\bunion\b.*\bselect\b',
        r'\bunion\b.*\ball\b.*\bselect\b',
        
        # Boolean-based injection
        r'\bor\b.*[\'"].*[\'"].*=.*[\'"].*[\'"]',
        r'\band\b.*[\'"].*[\'"].*=.*[\'"].*[\'"]',
        r'\bor\b.*\d+.*=.*\d+',
        r'\band\b.*\d+.*=.*\d+',
        
        # Time-based injection
        r'\bwaitfor\b.*\bdelay\b',
        r'\bsleep\b\s*\(',
        r'\bbenchmark\b\s*\(',
        
        # Error-based injection
        r'\bextractvalue\b\s*\(',
        r'\bupdatexml\b\s*\(',
        r'\bcast\b.*\bas\b.*\bint\b',
        
        # Stacked queries
        r';\s*(drop|delete|insert|update|create|alter|exec|execute)',
        
        # Comment injection
        r'/\*.*\*/',
        r'--\s',
        r'#.*',
        
        # Information schema attacks
        r'\binformation_schema\b',
        r'\bsys\b\.\b',
        r'\bmaster\b\.\b',
        
        # Function calls that might be dangerous
        r'\bload_file\b\s*\(',
        r'\binto\b.*\boutfile\b',
        r'\binto\b.*\bdumpfile\b',
        
        # Database-specific attacks
        r'\bpg_sleep\b\s*\(',  # PostgreSQL
        r'\brandomblob\b\s*\(',  # SQLite
        r'\buser\(\)',  # MySQL
        r'\bversion\(\)',
        r'\bdatabase\(\)',
        r'\bschema\(\)',
        
        # Hex/ASCII encoding attempts
        r'0x[0-9a-f]+',
        r'\bchar\b\s*\(\s*\d+\s*\)',
        r'\bascii\b\s*\(',
        
        # Conditional statements
        r'\bif\b\s*\([^)]*,',
        r'\bcase\b.*\bwhen\b.*\bthen\b',
        r'\biif\b\s*\(',  # SQL Server
    ]
    
    @classmethod
    def contains_sql_injection(cls, query_string: str) -> Tuple[bool, List[str]]:
        """
        Check if a string contains potential SQL injection patterns
        
        Args:
            query_string: String to check for SQL injection
            
        Returns:
            Tuple of (is_suspicious, list_of_matched_patterns)
        """
        if not query_string or not isinstance(query_string, str):
            return False, []
        
        # Normalize the query string
        normalized = query_string.lower().strip()
        matched_patterns = []
        
        # Check against known injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE | re.MULTILINE):
                matched_patterns.append(pattern)
        
        # Additional heuristics
        if cls._check_additional_heuristics(normalized):
            matched_patterns.append("additional_heuristics")
        
        return len(matched_patterns) > 0, matched_patterns
    
    @classmethod
    def _check_additional_heuristics(cls, query: str) -> bool:
        """Additional heuristic checks for SQL injection"""
        
        # Count of suspicious characters
        quote_count = query.count("'") + query.count('"')
        semicolon_count = query.count(';')
        equals_count = query.count('=')
        
        # Too many quotes might indicate injection
        if quote_count > 10:
            return True
        
        # Multiple statements (semicolons)
        if semicolon_count > 2:
            return True
        
        # Too many equality checks
        if equals_count > 5:
            return True
        
        # Suspicious keyword density
        sql_keywords = ['select', 'union', 'insert', 'update', 'delete', 'drop', 'create', 'alter']
        keyword_count = sum(1 for keyword in sql_keywords if keyword in query)
        if keyword_count > 3:
            return True
        
        return False

class SecureQueryBuilder:
    """Build secure SQL queries with parameterization"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def safe_execute(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Safely execute a raw SQL query with parameter binding
        
        Args:
            query: SQL query with named parameters (:param_name)
            params: Dictionary of parameters to bind
            
        Returns:
            Query result
            
        Raises:
            HTTPException: If query appears to contain SQL injection
        """
        # Check for SQL injection
        is_suspicious, patterns = SQLInjectionDetector.contains_sql_injection(query)
        if is_suspicious:
            logger.warning(f"Potential SQL injection detected. Patterns: {patterns}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query detected"
            )
        
        # Ensure parameters are provided for parameterized query
        if params is None:
            params = {}
        
        # Validate that all placeholders have corresponding parameters
        placeholders = re.findall(r':(\w+)', query)
        missing_params = [p for p in placeholders if p not in params]
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
        
        try:
            # Execute with parameter binding
            result = self.session.execute(text(query), params)
            return result
        except (SQLAlchemyError, DataError, IntegrityError) as e:
            logger.error(f"Database query error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database operation failed"
            )
    
    def safe_search(self, table: str, search_term: str, search_columns: List[str], 
                   additional_filters: Dict[str, Any] = None, limit: int = 50) -> Any:
        """
        Perform safe search operation with parameterized queries
        
        Args:
            table: Table name to search in
            search_term: Term to search for
            search_columns: List of column names to search in
            additional_filters: Additional WHERE conditions
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        # Validate table name (allow only alphanumeric and underscore)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
            raise ValueError("Invalid table name")
        
        # Validate column names
        for col in search_columns:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                raise ValueError(f"Invalid column name: {col}")
        
        # Sanitize search term
        search_term = search_term.replace('%', '\\%').replace('_', '\\_')
        
        # Build search conditions
        search_conditions = []
        params = {'search_term': f'%{search_term}%', 'limit': limit}
        
        for i, col in enumerate(search_columns):
            search_conditions.append(f"{col} ILIKE :search_term")
        
        where_clause = ' OR '.join(search_conditions)
        
        # Add additional filters
        if additional_filters:
            for key, value in additional_filters.items():
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                    raise ValueError(f"Invalid filter column: {key}")
                where_clause += f" AND {key} = :{key}"
                params[key] = value
        
        query = f"""
            SELECT * FROM {table}
            WHERE {where_clause}
            ORDER BY id DESC
            LIMIT :limit
        """
        
        return self.safe_execute(query, params)

class DatabaseSecurityMiddleware:
    """Middleware for database security monitoring"""
    
    def __init__(self):
        self.query_log = []
        self.max_log_size = 1000
        self.suspicious_query_count = 0
    
    def log_query(self, query: str, params: Dict = None, execution_time: float = None):
        """Log database query for security monitoring"""
        
        # Check for suspicious patterns
        is_suspicious, patterns = SQLInjectionDetector.contains_sql_injection(query)
        
        log_entry = {
            'timestamp': logging.Formatter().formatTime(logging.LogRecord(
                name='db_security', level=logging.INFO, pathname='', lineno=0,
                msg='', args=(), exc_info=None
            )),
            'query': query[:500],  # Truncate long queries
            'params_count': len(params) if params else 0,
            'execution_time': execution_time,
            'suspicious': is_suspicious,
            'patterns': patterns if is_suspicious else []
        }
        
        # Add to log
        self.query_log.append(log_entry)
        
        # Maintain log size
        if len(self.query_log) > self.max_log_size:
            self.query_log.pop(0)
        
        # Track suspicious queries
        if is_suspicious:
            self.suspicious_query_count += 1
            logger.warning(f"Suspicious query detected: {patterns}")
    
    def get_security_report(self) -> Dict:
        """Generate security report from logged queries"""
        total_queries = len(self.query_log)
        suspicious_queries = [q for q in self.query_log if q['suspicious']]
        
        return {
            'total_queries': total_queries,
            'suspicious_queries': len(suspicious_queries),
            'suspicious_percentage': (len(suspicious_queries) / total_queries * 100) if total_queries > 0 else 0,
            'recent_suspicious': suspicious_queries[-10:],  # Last 10 suspicious queries
            'most_common_patterns': self._get_common_patterns(suspicious_queries)
        }
    
    def _get_common_patterns(self, suspicious_queries: List[Dict]) -> Dict:
        """Get most common suspicious patterns"""
        pattern_count = {}
        for query in suspicious_queries:
            for pattern in query['patterns']:
                pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
        
        # Sort by frequency
        return dict(sorted(pattern_count.items(), key=lambda x: x[1], reverse=True))

class SecureORMHelper:
    """Helper for secure ORM operations"""
    
    @staticmethod
    def safe_filter(query: Query, filter_dict: Dict[str, Any]) -> Query:
        """
        Apply filters to SQLAlchemy query safely
        
        Args:
            query: SQLAlchemy Query object
            filter_dict: Dictionary of column: value filters
            
        Returns:
            Filtered query
        """
        for column, value in filter_dict.items():
            # Validate column name exists in the model
            if hasattr(query.column_descriptions[0]['type'], column):
                query = query.filter(getattr(query.column_descriptions[0]['type'], column) == value)
            else:
                logger.warning(f"Attempted to filter on non-existent column: {column}")
        
        return query
    
    @staticmethod
    def safe_order_by(query: Query, order_by: str, allowed_columns: List[str]) -> Query:
        """
        Apply ORDER BY clause safely
        
        Args:
            query: SQLAlchemy Query object
            order_by: Column name to order by (with optional - prefix for DESC)
            allowed_columns: List of allowed column names
            
        Returns:
            Ordered query
        """
        desc = False
        if order_by.startswith('-'):
            desc = True
            order_by = order_by[1:]
        
        if order_by not in allowed_columns:
            logger.warning(f"Attempted to order by non-allowed column: {order_by}")
            return query
        
        model_class = query.column_descriptions[0]['type']
        if hasattr(model_class, order_by):
            column = getattr(model_class, order_by)
            if desc:
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column)
        
        return query
    
    @staticmethod
    def safe_pagination(query: Query, page: int, size: int, max_size: int = 100) -> Tuple[Query, Dict]:
        """
        Apply pagination safely with limits
        
        Args:
            query: SQLAlchemy Query object
            page: Page number (1-based)
            size: Page size
            max_size: Maximum allowed page size
            
        Returns:
            Tuple of (paginated_query, pagination_info)
        """
        # Validate and sanitize pagination parameters
        page = max(1, int(page))
        size = max(1, min(int(size), max_size))
        
        # Calculate offset
        offset = (page - 1) * size
        
        # Apply pagination
        paginated_query = query.offset(offset).limit(size)
        
        # Calculate total count (expensive operation, use wisely)
        total_count = query.count()
        total_pages = (total_count + size - 1) // size
        
        pagination_info = {
            'page': page,
            'size': size,
            'total_count': total_count,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
        
        return paginated_query, pagination_info

# Global security middleware instance
db_security_middleware = DatabaseSecurityMiddleware()

def validate_table_name(table_name: str) -> bool:
    """Validate table name to prevent injection"""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name))

def validate_column_name(column_name: str) -> bool:
    """Validate column name to prevent injection"""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name))

def sanitize_like_pattern(pattern: str) -> str:
    """Sanitize LIKE pattern to prevent injection"""
    return pattern.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

def create_secure_database_session(database_url: str, **kwargs) -> Session:
    """
    Create a secure database session with security monitoring
    
    Args:
        database_url: Database connection URL
        **kwargs: Additional SQLAlchemy engine parameters
        
    Returns:
        Configured SQLAlchemy Session
    """
    # Set secure defaults
    secure_defaults = {
        'pool_pre_ping': True,  # Validate connections
        'pool_recycle': 3600,   # Recycle connections every hour
        'echo': False,          # Don't log SQL in production
        'future': True          # Use SQLAlchemy 2.0 style
    }
    
    # Merge with provided kwargs
    engine_kwargs = {**secure_defaults, **kwargs}
    
    # Create engine
    engine = create_engine(database_url, **engine_kwargs)
    
    # Create session
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return SessionLocal() 