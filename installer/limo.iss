; limo.iss
; Inno Setup Compiler Script for LIMO

[Setup]
AppId={{D1C083BE-2D12-4AB5-AA04-170BF1AE1CFD}
AppName=Local Intelligent Media Organizer (LIMO)
AppVersion=1.0.0
AppPublisher=Avijit Roy
AppPublisherURL=https://avijitroy.com
AppSupportURL=https://limo.avijitroy.com
AppUpdatesURL=https://limo.avijitroy.com
DefaultDirName={autopf}\LIMO
DefaultGroupName=LIMO
LicenseFile=license.txt
OutputDir=.
OutputBaseFilename=LIMO_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
DisableWelcomePage=no
SetupIconFile=logo.ico
WizardImageFile=wizard.bmp
WizardSmallImageFile=wizard_small.bmp
UninstallDisplayIcon={app}\logo.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "LIMO.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "logo.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\LIMO"; Filename: "{app}\LIMO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\logo.ico"; AppUserModelID: "com.avijitroy.limo"
Name: "{autodesktop}\LIMO"; Filename: "{app}\LIMO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\logo.ico"; AppUserModelID: "com.avijitroy.limo"; Tasks: desktopicon

[Run]
Filename: "{app}\LIMO.exe"; Description: "{cm:LaunchProgram,LIMO}"; Flags: nowait postinstall skipifsilent
