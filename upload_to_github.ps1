# GitHub Upload Script for Snatch-Alert Project
# This script will push your project to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GitHub Upload Script for Snatch-Alert" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get GitHub credentials
Write-Host "Please enter your GitHub credentials:" -ForegroundColor Yellow
$githubUsername = Read-Host "GitHub Username"
$githubToken = Read-Host "GitHub Personal Access Token (PAT)" -AsSecureString

# Convert secure string to plain text for git operations
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($githubToken)
$tokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Repository URL with credentials
$repoUrl = "https://${githubUsername}:${tokenPlain}@github.com/Salman17546/Snatch-Alert.git"

Write-Host ""
Write-Host "Starting upload process..." -ForegroundColor Green

# Check if .git exists in current directory, if so remove it to start fresh
if (Test-Path ".git") {
    Write-Host "Removing existing .git folder in root..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".git"
}

# Remove nested .git folders that might cause issues
$nestedGitFolders = @(
    "SnatchAlert/.git",
    "SnatchAlert/SnatchAlert/.git"
)

foreach ($gitFolder in $nestedGitFolders) {
    if (Test-Path $gitFolder) {
        Write-Host "Removing nested git folder: $gitFolder" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $gitFolder
    }
}

# Initialize new git repository
Write-Host "Initializing git repository..." -ForegroundColor Green
git init

# Configure git user (use GitHub username)
$userEmail = Read-Host "Enter your GitHub email"
git config user.name $githubUsername
git config user.email $userEmail

# Create .gitignore if it doesn't exist
if (-not (Test-Path ".gitignore")) {
    Write-Host "Creating .gitignore file..." -ForegroundColor Green
    @"
# Environment files
.env
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/

# Logs
*.log
logs/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Docker
*.tar

# Cache
*.cache
data/geocode_cache.json
"@ | Out-File -FilePath ".gitignore" -Encoding utf8
}

# Add all files
Write-Host "Adding files to git..." -ForegroundColor Green
git add -A

# Commit
Write-Host "Creating initial commit..." -ForegroundColor Green
git commit -m "Initial commit: Snatch-Alert project with Facebook Scraper integration"

# Add remote and push
Write-Host "Adding remote repository..." -ForegroundColor Green
git remote add origin $repoUrl

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Green
git branch -M main
$pushResult = git push -u origin main --force 2>&1

# Clean up - remove token from remote URL for security
git remote set-url origin "https://github.com/Salman17546/Snatch-Alert.git"

# Check if push was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Upload FAILED!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $pushResult" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Yellow
    Write-Host "1. Make sure your PAT has 'repo' scope enabled" -ForegroundColor White
    Write-Host "2. Check if the token has expired" -ForegroundColor White
    Write-Host "3. Verify the repository exists at:" -ForegroundColor White
    Write-Host "   https://github.com/Salman17546/Snatch-Alert" -ForegroundColor Cyan
    Write-Host "4. Make sure you're the owner or have write access" -ForegroundColor White
    Write-Host ""
    Write-Host "To create a new PAT with correct permissions:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "2. Click 'Generate new token (classic)'" -ForegroundColor White
    Write-Host "3. Check the 'repo' checkbox (full control)" -ForegroundColor White
    Write-Host "4. Click 'Generate token' and copy it" -ForegroundColor White
    Write-Host "5. Run this script again with the new token" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Upload Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your project has been uploaded to:" -ForegroundColor Cyan
Write-Host "https://github.com/Salman17546/Snatch-Alert" -ForegroundColor White
Write-Host ""
Write-Host "Note: If you don't have a Personal Access Token (PAT):" -ForegroundColor Yellow
Write-Host "1. Go to GitHub.com -> Settings -> Developer settings" -ForegroundColor White
Write-Host "2. Click 'Personal access tokens' -> 'Tokens (classic)'" -ForegroundColor White
Write-Host "3. Generate new token with 'repo' scope" -ForegroundColor White
