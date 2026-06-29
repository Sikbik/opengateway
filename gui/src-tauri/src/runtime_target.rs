#[cfg(any(target_os = "windows", test))]
pub(crate) fn windows_wsl_requested(
    bridge_flag: bool,
    distro_set: bool,
    wsl_workspace_set: bool,
    workspace: Option<&str>,
) -> bool {
    bridge_flag || distro_set || wsl_workspace_set || workspace.is_some_and(looks_like_linux_path)
}

#[cfg(any(target_os = "windows", test))]
pub(crate) fn windows_bundled_backend_preferred(
    debug_assertions: bool,
    opengateway_bin_set: bool,
) -> bool {
    !debug_assertions && !opengateway_bin_set
}

#[cfg(any(target_os = "windows", test))]
pub(crate) fn looks_like_linux_path(value: &str) -> bool {
    value.starts_with('/') || value.starts_with("~/")
}

#[cfg(any(target_os = "windows", test))]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WindowsRuntimeSelection {
    Wsl,
    UnavailableWsl,
    Bundled,
    Local,
}

#[cfg(any(target_os = "windows", test))]
pub(crate) fn select_windows_runtime(
    wsl_requested: bool,
    wsl_available: bool,
    bundled_preferred: bool,
) -> WindowsRuntimeSelection {
    if wsl_available {
        WindowsRuntimeSelection::Wsl
    } else if wsl_requested {
        WindowsRuntimeSelection::UnavailableWsl
    } else if bundled_preferred {
        WindowsRuntimeSelection::Bundled
    } else {
        WindowsRuntimeSelection::Local
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wsl_is_not_requested_by_default() {
        assert!(!windows_wsl_requested(false, false, false, None));
        assert!(!windows_wsl_requested(
            false,
            false,
            false,
            Some("C:\\Users\\cikbi\\project")
        ));
    }

    #[test]
    fn wsl_is_requested_by_explicit_flags_or_linux_workspace() {
        assert!(windows_wsl_requested(true, false, false, None));
        assert!(windows_wsl_requested(false, true, false, None));
        assert!(windows_wsl_requested(false, false, true, None));
        assert!(windows_wsl_requested(
            false,
            false,
            false,
            Some("/home/stache/project")
        ));
        assert!(windows_wsl_requested(
            false,
            false,
            false,
            Some("~/project")
        ));
    }

    #[test]
    fn bundled_backend_is_release_default_without_override() {
        assert!(windows_bundled_backend_preferred(false, false));
        assert!(!windows_bundled_backend_preferred(true, false));
        assert!(!windows_bundled_backend_preferred(false, true));
    }

    #[test]
    fn explicit_wsl_request_does_not_fall_back_when_bridge_is_unavailable() {
        assert_eq!(
            select_windows_runtime(true, false, true),
            WindowsRuntimeSelection::UnavailableWsl
        );
    }
}
