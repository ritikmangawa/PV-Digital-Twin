@echo off
echo ===================================================
echo   Starting Daily GitHub Update...
echo ===================================================

:: Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed or not in your PATH. Please install Git first.
    pause
    exit /b
)

:: Ensure a remote origin is set
git remote -v | find "origin" >nul
if %errorlevel% neq 0 (
    echo [ERROR] No remote repository found!
    echo Please link your GitHub repository first by running:
    echo git remote add origin https://github.com/yourusername/your-repo-name.git
    pause
    exit /b
)

:: Add all changes
echo [1/3] Staging files...
git add .

:: Commit with today's date
set timestamp=%date% %time%
echo [2/3] Committing changes...
git commit -m "Automated daily update: %timestamp%"

:: Push to main branch
echo [3/3] Pushing to GitHub...
git push origin main

echo ===================================================
echo   Update Complete!
echo ===================================================
pause
