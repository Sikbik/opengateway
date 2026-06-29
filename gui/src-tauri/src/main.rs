#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod control;
mod runtime_target;

fn main() {
    let builder = tauri::Builder::default();

    #[cfg(target_os = "windows")]
    let builder = builder.plugin(tauri_plugin_shell::init());

    builder
        .setup(|_app| {
            control::cleanup_managed_gateway_on_startup();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            control::load_snapshot,
            control::tail_logs,
            control::start_gateway,
            control::stop_gateway,
            control::run_doctor,
            control::run_login,
            control::probe_generation,
            control::probe_droid,
            control::sync_factory,
            control::set_droid_model,
        ])
        .build(tauri::generate_context!())
        .expect("error while building factory-control")
        .run(|_app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                control::stop_managed_gateway_on_exit();
            }
        });
}
