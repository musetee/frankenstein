@echo off
REM Get current date in YYYY-MM-DD format
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set date=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%

REM Add, commit, and push changes in the parent directory

git add .
git commit -m "%date%"
git push

echo Script completed.
