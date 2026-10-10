; Inno Setup script for the Ask Physics Windows installer (ADR-023).
;
; Build (the release workflow does this; Inno Setup 6.3 or newer):
;
;     ISCC.exe /DAppVersion=0.4.0 /O"dist" packaging\windows\askphysics.iss
;
; That writes dist\AskPhysicsSetup-0.4.0.exe. The installer is per-user (no administrator
; prompt) and silent-capable:
;
;     AskPhysicsSetup-0.4.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
;
; It does not carry the program. It copies install.ps1 and a small runner into the app
; folder and, at the end of setup, runs install.ps1 pinned to this release (ASKPHYSICS_REF),
; which installs uv if needed, runs `uv tool install askphysics`, and downloads the models.
; Set ASKPHYSICS_SOURCE in the environment to install from a local checkout instead (the
; smoke tests do). The uninstaller runs `uv tool uninstall askphysics`. Downloaded models in
; %USERPROFILE%\.cache\askphysics\models are left alone: they are the user's data, a reinstall
; reuses them, and pip or brew installs of Ask Physics share the same folder.
;
; AppId is the identity of the program. It must never change: WinGet, Chocolatey and
; Windows upgrade and uninstall matching all rely on it. The registry key it produces is
; "{007795BD-A0E0-46D3-8C03-CE338CD7B57D}_is1", which the WinGet manifest lists as ProductCode.

#ifndef AppVersion
  #error AppVersion is not defined. Build with: ISCC.exe /DAppVersion=X.Y.Z askphysics.iss
#endif

[Setup]
AppId={{007795BD-A0E0-46D3-8C03-CE338CD7B57D}
AppName=Ask Physics
AppVersion={#AppVersion}
AppVerName=Ask Physics {#AppVersion}
AppPublisher=Sachin Shankar
AppPublisherURL=https://askphysics.vercel.app
AppSupportURL=https://github.com/shankar-sachin/ask-physics/issues
AppUpdatesURL=https://github.com/shankar-sachin/ask-physics/releases
VersionInfoVersion={#AppVersion}
VersionInfoDescription=Ask Physics installer
UninstallDisplayName=Ask Physics
; Per user, never elevated: with "lowest", {autopf} is %LOCALAPPDATA%\Programs.
PrivilegesRequired=lowest
DefaultDirName={autopf}\Ask Physics
DisableDirPage=yes
DisableProgramGroupPage=yes
; Windows on ARM runs this through x64 emulation; Setup then uses the native System32.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
OutputBaseFilename=AskPhysicsSetup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; uv adds a folder to the user's PATH; tell Explorer and new terminals.
ChangesEnvironment=yes
CloseApplications=no
RestartApplications=no

[Messages]
FinishedLabelNoIcons=Setup has finished installing [name].%n%nOpen a new terminal, so it picks up the updated PATH, and run:%n%n    askphysics ask "A ball is dropped from 20 m. How fast does it land?"

[Files]
Source: "..\..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "askphysics-setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Code]
const
  AppReleaseTag = 'v{#AppVersion}';

// Where the runner writes what install.ps1 and uv printed.
function SetupLogPath: String;
begin
  Result := ExpandConstant('{%TEMP}\AskPhysics-setup.log');
end;

// The last Count lines of a text file, or '' when it can't be read.
function LogTail(const FileName: String; Count: Integer): String;
var
  Lines: TArrayOfString;
  I, First: Integer;
begin
  Result := '';
  if LoadStringsFromFile(FileName, Lines) then
  begin
    First := GetArrayLength(Lines) - Count;
    if First < 0 then
      First := 0;
    for I := First to GetArrayLength(Lines) - 1 do
      Result := Result + Lines[I] + #13#10;
  end;
end;

// Runs askphysics-setup.ps1 (install or uninstall) with the 64-bit Windows PowerShell,
// hidden, and waits for it. Returns False when PowerShell could not be started.
function RunSetupScript(const Action: String; var ResultCode: Integer): Boolean;
var
  Params: String;
begin
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\askphysics-setup.ps1') + '"' +
    ' -Action ' + Action + ' -Ref "' + AppReleaseTag + '" -Log "' + SetupLogPath + '"';
  Log('Running: powershell.exe ' + Params);
  Result := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, '',
    SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Msg: String;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardSilent then
      WizardForm.StatusLabel.Caption :=
        'Installing Ask Physics ' + AppReleaseTag + ' (downloads Python, the packages and the models; this can take several minutes)...';
    if not RunSetupScript('install', ResultCode) then
    begin
      Msg := 'Ask Physics could not be installed: Windows PowerShell could not be started (' +
        SysErrorMessage(ResultCode) + ').';
      Log(Msg);
      SuppressibleMsgBox(Msg, mbCriticalError, MB_OK, IDOK);
      RaiseException(Msg);
    end
    else if ResultCode <> 0 then
    begin
      Msg := 'Ask Physics could not be installed: install.ps1 exited with code ' + IntToStr(ResultCode) +
        '.' + #13#10#13#10 + LogTail(SetupLogPath, 12) + #13#10 + 'Full output: ' + SetupLogPath;
      Log(Msg);
      SuppressibleMsgBox(Msg, mbCriticalError, MB_OK, IDOK);
      RaiseException('Ask Physics install.ps1 failed with exit code ' + IntToStr(ResultCode) + '; see ' + SetupLogPath);
    end;
    Log('install.ps1 finished with exit code 0');
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    // `uv tool uninstall askphysics`; the script ignores a missing uv or a tool that is
    // already gone, so removing the program never fails on this.
    if RunSetupScript('uninstall', ResultCode) then
      Log('Uninstall script finished with exit code ' + IntToStr(ResultCode))
    else
      Log('Could not start PowerShell to run uv tool uninstall; skipped');
  end;
end;
