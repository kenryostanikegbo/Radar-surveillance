@echo off
REM Build the radar_parse.dll (C firmware) so the Python loopback test
REM can exercise the exact same parser/feature-extractor the ESP32-S3
REM firmware will use.
REM
REM Output: C:\tmp\radar_parse.dll
REM
REM Exposes:
REM     radar_crc16(data, len)                       -> uint16
REM     radar_parse_frame(buf, len, out_targets, ts) -> int
REM     radar_frame_size()                           -> int (66)
REM     feature_extract_window(window, mask, out)    -> int (n_features)
REM     feature_normalize(x)                         -> void
REM     feature_quantize(in, out, n)                 -> void
REM     tracker_init(t)                              -> void
REM     tracker_step(t, dt, dets, n)                 -> void
REM     int_softmax(logits, probs, n)                -> void
REM     dashboard_state_json(buf, len, ...)          -> int
REM     dashboard_health_json(buf, len, ...)         -> int

setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo Failed to set up MSVC env
    exit /b 1
)
cd /d C:\Users\HomePC\Documents\RadarSurveillance\phase2
mkdir C:\tmp 2>nul
del /Q C:\tmp\radar_parse.dll C:\tmp\parse_frame.obj C:\tmp\feature_extract.obj C:\tmp\tracker.obj C:\tmp\inference.obj C:\tmp\dashboard_http.obj 2>nul

REM Compile sources (include firmware/ + models/ for headers)
set CFLAGS=/nologo /c /I firmware /I models /O2

cl %CFLAGS% /Fo:C:\tmp\parse_frame.obj firmware\parse_frame.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /Fo:C:\tmp\feature_extract.obj firmware\feature_extract.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /Fo:C:\tmp\tracker.obj firmware\tracker.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /Fo:C:\tmp\inference.obj firmware\inference.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /Fo:C:\tmp\dashboard_http.obj firmware\dashboard_http.c
if errorlevel 1 exit /b 1

REM Link into one DLL
link /nologo /DLL /DEF:firmware\radar_parse.def /MACHINE:X64 /OUT:C:\tmp\radar_parse.dll ^
     C:\tmp\parse_frame.obj C:\tmp\feature_extract.obj C:\tmp\tracker.obj C:\tmp\inference.obj C:\tmp\dashboard_http.obj
if errorlevel 1 exit /b 1
echo Built C:\tmp\radar_parse.dll
endlocal
