# This script is used to migrate files to Git LFS

# Check if Git LFS is installed
if ! git lfs version &>/dev/null; then
    echo "Git LFS is not installed. Please install it first."
    exit 1
fi

# Migrate files to LFS
# Add your migration commands here

# Example command to migrate specific files
# git lfs migrate import --include="*.bin,*.pt"

# Add any additional migration logic as needed 