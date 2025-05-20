# run this script to check for changes in the ./src directory
if git diff --cached --quiet -- ./src; then
  echo "No changes detected in ./src"
else
  echo "Changes detected in ./src"
  task commit:prepare:src
fi
