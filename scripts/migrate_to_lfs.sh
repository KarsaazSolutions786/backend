#!/usr/bin/env bash
# One-time history rewrite: converts existing *.bin|*.onnx|*.safetensors|*.lock blobs to LFS
set -euo pipefail

echo "🚀 Starting Git LFS migration for ML model artifacts..."

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS not found. Please install it first:"
    echo "   macOS: brew install git-lfs"
    echo "   Ubuntu: sudo apt-get install git-lfs"
    echo "   Windows: https://git-lfs.github.com/"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Check if we have uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

patterns='*.bin,*.onnx,*.safetensors,*.lock,*.pt,*.pth,models/**/blobs/*,models/**/snapshots/*/blobs/*'

echo "📋 Installing Git LFS..."
git lfs install

echo "🔄 Migrating existing files to LFS..."
echo "   Patterns: $patterns"
git lfs migrate import --include="$patterns"

echo "✅ History rewritten successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Force push to update the remote repository:"
echo "   git push --force-with-lease origin \$(git branch --show-current)"
echo ""
echo "2. Verify LFS tracking:"
echo "   git lfs ls-files"
echo ""
echo "⚠️  Warning: This rewrites git history. Make sure all collaborators"
echo "   are aware and have backed up their work." 