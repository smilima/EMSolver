@echo off
call "C:\Program Files (x86)\Embarcadero\Studio\37.0\bin\rsvars.bat"
msbuild "C:\Users\andre\OneDrive\Documents\CCode\2026\RFSimulator\RFSimulator.cbproj" /t:Build /p:Config=Debug /p:Platform=Win64x /nologo /v:minimal
