@echo off
setlocal

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%"

py -3.11 -m uvicorn api.app:app --host 0.0.0.0 --port 8000

popd
endlocal
