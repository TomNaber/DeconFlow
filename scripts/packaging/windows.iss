[Setup]
AppId={{BACFB965-46BD-42C8-B0F9-43F89742407B}
AppName=Batch Deconvolution
AppVersion={#AppVersion}
VersionInfoVersion={#NativeVersion}
DefaultDirName={localappdata}\Programs\Batch Deconvolution
DefaultGroupName=Batch Deconvolution
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Batch Deconvolution

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Batch Deconvolution"; Filename: "{app}\Launch.cmd"; WorkingDir: "{app}"

[Run]
Filename: "{app}\Launch.cmd"; Description: "Open Batch Deconvolution"; Flags: postinstall shellexec skipifsilent
