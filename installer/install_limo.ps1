# install_limo.ps1
# Native Windows Installer Script for LIMO

$AppName = "LIMO"
$AppDir = "$env:LOCALAPPDATA\LIMO"
$ExeName = "LIMO.exe"
$IconName = "logo.ico"
$SrcExe = Join-Path $PSScriptRoot $ExeName
$SrcIcon = Join-Path $PSScriptRoot $IconName
$DestExe = Join-Path $AppDir $ExeName
$DestIcon = Join-Path $AppDir $IconName

# Make sure console output is visible and clean
Clear-Host
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Installing Local Intelligent Media Organizer   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Create App Directory
if (-not (Test-Path $AppDir)) {
    New-Item -ItemType Directory -Path $AppDir -Force | Out-Null
}

# 2. Copy Executable
if (-not (Test-Path $SrcExe)) {
    Write-Host "Error: LIMO.exe not found in source directory." -ForegroundColor Red
    Write-Host "Please make sure LIMO.exe is in the same folder as this installer script." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    Exit 1
}

Write-Host "Copying application files..." -ForegroundColor Gray
Copy-Item -Path $SrcExe -Destination $DestExe -Force
if (Test-Path $SrcIcon) {
    Copy-Item -Path $SrcIcon -Destination $DestIcon -Force
}
$ShortcutIcon = if (Test-Path $DestIcon) { $DestIcon } else { $DestExe }

# 3. Create Uninstaller Script inside the AppDir
Write-Host "Setting up uninstaller registry hooks..." -ForegroundColor Gray
$UninstallScriptPath = Join-Path $AppDir "uninstall.ps1"
$UninstallContent = @"
# uninstall_limo.ps1
Clear-Host
Write-Host "==================================================" -ForegroundColor Red
Write-Host "          Uninstalling LIMO Organizer            " -ForegroundColor Red
Write-Host "==================================================" -ForegroundColor Red
Write-Host ""

# Remove Start Menu shortcut
`$StartMenuLnk = "`$env:APPDATA\Microsoft\Windows\Start Menu\Programs\LIMO.lnk"
if (Test-Path `$StartMenuLnk) {
    Write-Host "Removing Start Menu shortcut..." -ForegroundColor Gray
    Remove-Item `$StartMenuLnk -Force
}

# Remove Desktop shortcut
`$DesktopLnk = Join-Path ([System.Environment]::GetFolderPath("Desktop")) "LIMO.lnk"
if (Test-Path `$DesktopLnk) {
    Write-Host "Removing Desktop shortcut..." -ForegroundColor Gray
    Remove-Item `$DesktopLnk -Force
}

# Delete Registry Entries
`$RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LIMO"
if (Test-Path `$RegPath) {
    Write-Host "Cleaning Windows registry entries..." -ForegroundColor Gray
    Remove-Item `$RegPath -Force
}

Write-Host "Removing installation folder..." -ForegroundColor Gray
`$AppDir = "`$env:LOCALAPPDATA\LIMO"
if (Test-Path `$AppDir) {
    # Remove files but cannot delete self while running, so we delete it via a deferred PowerShell background task
    Get-ChildItem `$AppDir | Where-Object { `$_.Name -ne "uninstall.ps1" } | Remove-Item -Force
    Start-Process powershell -ArgumentList "-NoProfile -Command `"Start-Sleep -Seconds 1; Remove-Item -Path '$AppDir' -Recurse -Force`"" -WindowStyle Hidden
}

Write-Host ""
Write-Host "LIMO has been successfully uninstalled from your computer." -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close this window..."
"@
Set-Content -Path $UninstallScriptPath -Value $UninstallContent -Force

# 4. Create Shortcuts (Desktop & Start Menu)
Write-Host "Creating Start Menu and Desktop shortcuts..." -ForegroundColor Gray
$WshShell = New-Object -ComObject WScript.Shell

# Start Menu Shortcut
$StartMenuDir = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
$StartMenuLnk = Join-Path $StartMenuDir "LIMO.lnk"
$Shortcut = $WshShell.CreateShortcut($StartMenuLnk)
$Shortcut.TargetPath = $DestExe
$Shortcut.WorkingDirectory = $AppDir
$Shortcut.IconLocation = $ShortcutIcon
$Shortcut.Description = "Local Intelligent Media Organizer"
$Shortcut.Save()

# Desktop Shortcut
$DesktopDir = [System.Environment]::GetFolderPath("Desktop")
$DesktopLnk = Join-Path $DesktopDir "LIMO.lnk"
$ShortcutDesktop = $WshShell.CreateShortcut($DesktopLnk)
$ShortcutDesktop.TargetPath = $DestExe
$ShortcutDesktop.WorkingDirectory = $AppDir
$ShortcutDesktop.IconLocation = $ShortcutIcon
$ShortcutDesktop.Description = "Local Intelligent Media Organizer"
$ShortcutDesktop.Save()

# 5. Register in Windows Add/Remove Programs (Registry)
$RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LIMO"
if (-not (Test-Path $RegPath)) {
    New-Item -Path $RegPath -Force | Out-Null
}

Set-ItemProperty -Path $RegPath -Name "DisplayName" -Value "Local Intelligent Media Organizer (LIMO)" -Force
Set-ItemProperty -Path $RegPath -Name "UninstallString" -Value "powershell.exe -ExecutionPolicy Bypass -File `"$UninstallScriptPath`"" -Force
Set-ItemProperty -Path $RegPath -Name "DisplayIcon" -Value $ShortcutIcon -Force
Set-ItemProperty -Path $RegPath -Name "Publisher" -Value "Avijit Roy" -Force
Set-ItemProperty -Path $RegPath -Name "DisplayVersion" -Value "1.0.0" -Force
Set-ItemProperty -Path $RegPath -Name "HelpLink" -Value "https://limo.avijitroy.com" -Force

Write-Host ""
Write-Host "SUCCESS: LIMO has been successfully installed!" -ForegroundColor Green
Write-Host "You can find shortcuts on your Desktop and in the Start Menu." -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close this installer..."
