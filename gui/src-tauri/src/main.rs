mod control;

fn main() {
    let builder = tauri::Builder::default();

    #[cfg(target_os = "windows")]
    let builder = builder.plugin(tauri_plugin_shell::init());

    builder
        .invoke_handler(tauri::generate_handler![
            control::load_snapshot,
            control::tail_logs,
            control::start_gateway,
            control::stop_gateway,
            control::run_doctor,
            control::sync_factory,
            control::set_droid_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running factory-control");
}
