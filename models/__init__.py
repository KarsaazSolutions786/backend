# Models package
# Import all models to make them available at package level
from .models import *

# Also make the models module available for 'from models.models import' pattern
from . import models 