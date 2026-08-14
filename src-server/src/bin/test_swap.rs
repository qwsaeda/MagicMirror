use magic_server::inference::TinyFace;
use magic_server::inference::detector::ImageData;

fn main() {
    let models_dir = std::path::PathBuf::from(r"C:\Users\Administrator\MagicMirror\models");
    let mut tiny = TinyFace::new();
    tiny.load_models(&models_dir).unwrap();
    tiny.prepare().unwrap();

    // Use paths relative to the current working directory (project root when run from src-server)
    let input = std::fs::read(r"..\tests\fixtures\a.jpg")
        .or_else(|_| std::fs::read(r"tests\fixtures\a.jpg"))
        .unwrap();
    let target = std::fs::read(r"..\tests\fixtures\b.png")
        .or_else(|_| std::fs::read(r"tests\fixtures\b.png"))
        .unwrap();

    let src_boxes = tiny.detect_faces(&input).unwrap();
    let tgt_boxes = tiny.detect_faces(&target).unwrap();
    println!("Src faces: {}", src_boxes.len());
    println!("Tgt faces: {}", tgt_boxes.len());

    if src_boxes.is_empty() || tgt_boxes.is_empty() {
        println!("No faces");
        return;
    }

    let sb = &src_boxes[0];
    let tb = &tgt_boxes[0];
    println!("Src score={:.3} bbox=({:.1},{:.1},{:.1},{:.1})", sb.score, sb.x1, sb.y1, sb.x2, sb.y2);
    println!("Src landmark: {:?}", sb.landmarks);
    println!("Tgt score={:.3} bbox=({:.1},{:.1},{:.1},{:.1})", tb.score, tb.x1, tb.y1, tb.x2, tb.y2);
    println!("Tgt landmark: {:?}", tb.landmarks);

    // Try swap
    match tiny.swap_face(&input, &sb, &target, &tb) {
        Ok(result) => {
            let id = ImageData::from_bytes(&input).unwrap();
            println!("Input size: {}x{}", id.width, id.height);
            println!("Result size: {} bytes (expect {}x{}x3 = {})", result.len(), id.width, id.height, id.width*id.height*3);
            // Check if result differs from input image
            let img = magic_server::inference::detector::ImageData::from_bytes(&input).unwrap();
            let rgb = img.to_rgb();
            let mut diff: u64 = 0;
            let n = result.len().min(rgb.data.len());
            for i in 0..n {
                diff += (result[i] as i32 - rgb.data[i] as i32).unsigned_abs() as u64;
            }
            println!("Result mean diff from input: {:.2}", diff as f64 / n as f64);
            std::fs::write(r"C:\Users\Administrator\AppData\Local\Temp\opencode\rust_result.jpg", &result).unwrap();
        }
        Err(e) => println!("ERROR: {:?}", e),
    }
}