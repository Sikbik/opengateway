!macro NSIS_HOOK_PREINSTALL
  DetailPrint "Stopping running Factory Control processes..."
  nsExec::ExecToLog 'taskkill /F /T /IM factory-control.exe'
  nsExec::ExecToLog 'taskkill /F /T /IM opengateway.exe'
  nsExec::ExecToLog 'taskkill /F /T /IM opengateway-runtime.exe'
  Sleep 1200
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  DetailPrint "Stopping running Factory Control processes..."
  nsExec::ExecToLog 'taskkill /F /T /IM factory-control.exe'
  nsExec::ExecToLog 'taskkill /F /T /IM opengateway.exe'
  nsExec::ExecToLog 'taskkill /F /T /IM opengateway-runtime.exe'
  Sleep 1200
!macroend
