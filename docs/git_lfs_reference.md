# Git LFS Reference Guide

This document provides comprehensive instructions for installing, configuring, and removing Git Large File Storage (LFS) from repositories.

## 📋 Table of Contents

- [What is Git LFS](#what-is-git-lfs)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)
- [Best Practices](#best-practices)

## 🔍 What is Git LFS

Git Large File Storage (LFS) is an extension that replaces large files with text pointers inside Git, while storing the file contents on a remote server. This is useful for:

- Large binary files (images, videos, models)
- Files that change frequently and are large
- Keeping repository size manageable
- Faster clone and fetch operations

**When to use LFS:**
- Image files > 1MB
- Model files > 10MB
- Video/audio files
- Large datasets

**When NOT to use LFS:**
- Small repositories
- Primarily text-based projects
- When you need offline access to all files
- Simple workflows without large files

## 📦 Installation

### macOS (Homebrew)
```bash
# Install Git LFS
brew install git-lfs

# Verify installation
git lfs version
```

### Ubuntu/Debian
```bash
# Install Git LFS
sudo apt-get update
sudo apt-get install git-lfs

# Verify installation
git lfs version
```

### CentOS/RHEL
```bash
# Install Git LFS
sudo yum install git-lfs

# Verify installation
git lfs version
```

### Conda Environment
```bash
# Install Git LFS via conda
conda install -c conda-forge git-lfs

# Verify installation
git lfs version
```

### Manual Installation
```bash
# Download from GitHub releases
curl -s https://api.github.com/repos/git-lfs/git-lfs/releases/latest \
  | grep "browser_download_url.*linux-amd64" \
  | cut -d : -f 2,3 \
  | tr -d \" \
  | wget -qi -

# Extract and install
tar -xzf git-lfs-*.tar.gz
sudo ./git-lfs-*/install.sh
```

## ⚙️ Configuration

### Initialize Git LFS in Repository
```bash
# Navigate to your repository
cd /path/to/your/repo

# Initialize Git LFS (one-time setup)
git lfs install

# This creates hooks in .git/hooks/
```

### Configure File Tracking
```bash
# Track specific file types
git lfs track "*.png"
git lfs track "*.jpg"
git lfs track "*.mp4"
git lfs track "*.zip"

# Track files in specific directories
git lfs track "models/*"
git lfs track "datasets/*.csv"

# Track files by size (files > 100MB)
git lfs track "*.bin" --lockable

# View tracked patterns
git lfs track
```

### Commit LFS Configuration
```bash
# Add the .gitattributes file (created by git lfs track)
git add .gitattributes

# Commit the LFS configuration
git commit -m "Configure Git LFS for large files"

# Push to remote
git push origin main
```

## 🚀 Usage

### Adding Files to LFS
```bash
# Add large files (will be tracked if patterns match)
git add large_model.bin
git add datasets/images/*.png

# Commit and push
git commit -m "Add large model and images"
git push origin main
```

### Checking LFS Status
```bash
# View LFS files in current commit
git lfs ls-files

# View LFS status
git lfs status

# View LFS environment info
git lfs env
```

### Migrating Existing Files to LFS
```bash
# Migrate existing files to LFS (rewrites history)
git lfs migrate import --include="*.png" --everything

# Migrate specific files
git lfs migrate import --include="models/large_model.bin"

# Force push after migration (rewrites history)
git push --force-with-lease origin main
```

### Downloading LFS Files
```bash
# Download all LFS files
git lfs pull

# Download specific LFS files
git lfs pull --include="*.png"
git lfs pull --exclude="*.zip"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. LFS Not Found Error
```bash
# Error: git-lfs: not found
# Solution: Install Git LFS
brew install git-lfs  # macOS
sudo apt-get install git-lfs  # Ubuntu

# Then initialize
git lfs install
```

#### 2. Files Not Tracked by LFS
```bash
# Check if files are tracked
git lfs ls-files

# Check tracking patterns
git lfs track

# Add tracking pattern if missing
git lfs track "*.png"
git add .gitattributes
git commit -m "Update LFS tracking"
```

#### 3. Push Failures
```bash
# Error: failed to push some refs
# Solution: Check LFS configuration
git lfs env

# Verify remote supports LFS
git lfs ls-files --all
```

#### 4. Large Repository Size
```bash
# Check what's taking space
git lfs ls-files --size

# Migrate more files to LFS
git lfs migrate import --include="*.zip" --everything
```

#### 5. Clone Issues
```bash
# Clone without LFS files (faster)
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>

# Download LFS files after clone
git lfs pull
```

### Performance Tips
```bash
# Skip LFS download during clone
git clone --filter=blob:none <repo-url>

# Download only recent LFS files
git lfs pull --recent

# Exclude large files from pull
git lfs pull --exclude="*.zip"
```

## 🔥 Uninstallation

### Complete LFS Removal Process

#### Step 1: Remove LFS Tracking
```bash
# Remove all LFS tracking patterns
rm -f .gitattributes

# Or manually edit .gitattributes to remove LFS patterns
# Remove lines like: *.png filter=lfs diff=lfs merge=lfs -text
```

#### Step 2: Migrate Files Back to Git
```bash
# Migrate LFS files back to regular Git (rewrites history)
git lfs migrate export --include="*.png" --everything

# Or migrate all LFS files
git lfs migrate export --everything
```

#### Step 3: Uninstall LFS from Repository
```bash
# Remove LFS hooks from repository
git lfs uninstall

# This removes hooks from .git/hooks/
```

#### Step 4: Clean Up and Commit
```bash
# Stage the removal of .gitattributes
git add .gitattributes

# Commit the changes
git commit -m "Remove Git LFS configuration"

# Force push (if you migrated files)
git push --force-with-lease origin main
```

#### Step 5: Verify Removal
```bash
# Check no LFS files remain
git lfs ls-files

# Should show: (empty)

# Check repository size
du -sh .git/
```

### Alternative: Keep Files in LFS but Disable Locally
```bash
# Just remove LFS hooks (files stay in LFS)
git lfs uninstall

# Remove .gitattributes if you don't want new files tracked
rm .gitattributes
git add .gitattributes
git commit -m "Disable LFS tracking"
```

### Remove LFS Software (Optional)
```bash
# macOS
brew uninstall git-lfs

# Ubuntu/Debian
sudo apt-get remove git-lfs

# Conda
conda remove git-lfs
```

## 📚 Best Practices

### 1. File Selection
```bash
# Good candidates for LFS
git lfs track "*.png"      # Images
git lfs track "*.jpg"      # Images
git lfs track "*.mp4"      # Videos
git lfs track "*.zip"      # Archives
git lfs track "*.bin"      # Binary files
git lfs track "*.h5"       # Model files
git lfs track "*.pkl"      # Pickle files

# Avoid LFS for
# - Small files < 1MB
# - Frequently changing text files
# - Files you need offline access to
```

### 2. Repository Structure
```
project/
├── .gitattributes          # LFS configuration
├── src/                    # Source code (regular Git)
├── docs/                   # Documentation (regular Git)
├── models/                 # Large model files (LFS)
│   └── *.bin
├── datasets/               # Large datasets (LFS)
│   └── *.csv
└── media/                  # Images/videos (LFS)
    ├── *.png
    └── *.mp4
```

### 3. Workflow Best Practices
```bash
# Always check LFS status before committing
git lfs status

# Use descriptive commit messages for LFS files
git commit -m "Add trained model v2.1 (45MB)"

# Regularly clean up old LFS files
git lfs prune

# Use .gitignore for temporary large files
echo "*.tmp" >> .gitignore
echo "temp_data/" >> .gitignore
```

### 4. Team Collaboration
```bash
# Document LFS setup in README
echo "## Setup" >> README.md
echo "This project uses Git LFS for large files." >> README.md
echo "Install: brew install git-lfs" >> README.md
echo "Setup: git lfs install" >> README.md

# Include .gitattributes in first commit
git add .gitattributes
git commit -m "Initial LFS configuration"
```

### 5. Performance Optimization
```bash
# Configure LFS for better performance
git config lfs.batch true
git config lfs.concurrenttransfers 10

# Use LFS with shallow clones
git clone --depth=1 <repo-url>
git lfs pull
```

## 🔍 Examples

### Example 1: Setting up LFS for ML Project
```bash
# Initialize LFS
git lfs install

# Track model files
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "*.joblib"

# Track dataset files
git lfs track "data/*.csv"
git lfs track "data/*.parquet"

# Track checkpoints
git lfs track "checkpoints/*"

# Commit configuration
git add .gitattributes
git commit -m "Configure LFS for ML project"
```

### Example 2: Migrating Existing Repository
```bash
# Migrate existing large files
git lfs migrate import --include="*.png,*.jpg" --everything

# Add new tracking patterns
git lfs track "*.mp4"
git lfs track "models/*.bin"

# Commit and push
git add .gitattributes
git commit -m "Migrate to LFS and add new patterns"
git push --force-with-lease origin main
```

### Example 3: Selective LFS Download
```bash
# Clone without LFS files
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
cd repo

# Download only specific file types
git lfs pull --include="*.png"

# Or exclude large files
git lfs pull --exclude="*.zip"
```

## 📖 Additional Resources

- [Git LFS Official Documentation](https://git-lfs.github.io/)
- [GitHub LFS Documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS Tutorial](https://www.atlassian.com/git/tutorials/git-lfs)
- [Git LFS Cheat Sheet](https://github.com/git-lfs/git-lfs/wiki/Tutorial)

## 🎯 Quick Reference

### Installation Commands
```bash
# Install
brew install git-lfs        # macOS
sudo apt install git-lfs    # Ubuntu
conda install -c conda-forge git-lfs  # Conda

# Initialize
git lfs install
```

### Configuration Commands
```bash
git lfs track "*.png"       # Track file type
git lfs track              # List tracked patterns
git lfs ls-files           # List LFS files
git lfs status             # Check LFS status
```

### Removal Commands
```bash
git lfs uninstall          # Remove LFS hooks
rm .gitattributes          # Remove LFS configuration
git lfs migrate export --everything  # Migrate files back
```

### Troubleshooting Commands
```bash
git lfs env                # Check LFS environment
git lfs pull               # Download LFS files
git lfs prune              # Clean up old files
git lfs version            # Check LFS version
```

---

*This guide covers Git LFS management for development workflows. Always test LFS changes in a separate branch before applying to main.*