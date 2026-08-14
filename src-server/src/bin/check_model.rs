use magic_server::inference::OnnxModel;
use magic_server::inference::swapper::Swapper;

fn main() {
    let models_dir = std::path::PathBuf::from(r"C:\Users\Administrator\MagicMirror\models");
    
    let mut swapper = Swapper::new();
    swapper.load(models_dir.join("inswapper_128_fp16.onnx")).unwrap();
    swapper.prepare().unwrap();

    // Try to access session and print input/output info
    println!("Swapper model loaded successfully");
}