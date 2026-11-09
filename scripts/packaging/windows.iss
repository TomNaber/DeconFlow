[Setup]
AppId={{BACFB965-46BD-42C8-B0F9-43F89742407B}
AppName=DeconFlow
AppPublisher=Tom Naber
AppPublisherURL=https://github.com/TomNaber/DeconFlow
AppSupportURL=https://github.com/TomNaber/DeconFlow/issues
AppVersion={#AppVersion}
VersionInfoVersion={#NativeVersion}
DefaultDirName={localappdata}\Programs\DeconFlow
DefaultGroupName=DeconFlow
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=DeconFlow

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DeconFlow"; Filename: "{app}\Launch.cmd"; WorkingDir: "{app}"

[Run]
Filename: "{app}\Launch.cmd"; Description: "Open DeconFlow"; Flags: postinstall shellexec skipifsilent
