use anyhow::{Result, Context};
use tch::{Tensor, Kind, CModule};
use std::env;
/// Face classifier using a TorchScript ResNet50 model
struct ResNet50FaceClassifier {
    model: CModule,
}

impl ResNet50FaceClassifier {
    fn new() -> Result<Self> {
        // Load complete TorchScript model 
        let model = CModule::load("models/resnet50/full_resnet50_cpu.pt")
            .context("Failed to load full model")?;
        
        println!("loaded ResNet50 full model");
        
        Ok(ResNet50FaceClassifier { model })
    }
    
    fn predict(&self, image_path: &str) -> Result<(String, f32)> {
        // Load and resize image to 224x224 (ResNet input size)
        let img = image::open(image_path)
            .context("Failed to open image")?
            .resize_exact(224, 224, image::imageops::FilterType::Lanczos3)
            .to_rgb8();

        // Convert pixels to tensor
        let pixels: Vec<f32> = img.pixels()
            .flat_map(|p| vec![p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        
        let tensor = Tensor::from_slice(&pixels)
            .view([224, 224, 3])
            .permute(&[2, 0, 1])
            / 255.0;
        
        // ImageNet normalization
        let mean = Tensor::from_slice(&[0.485_f32, 0.456, 0.406]).view([3, 1, 1]);
        let std = Tensor::from_slice(&[0.229_f32, 0.224, 0.225]).view([3, 1, 1]);
        let normalized = (tensor - mean) / std;
        
        let batch = normalized.unsqueeze(0);
        
        let output = tch::no_grad(|| {
            self.model.forward_ts(&[batch]).unwrap()
        });
        
        let probs = output.softmax(-1, Kind::Float);
        let pred_idx = output.argmax(-1, false).int64_value(&[]);
        
        let confidence = probs.double_value(&[0, pred_idx]) as f32 * 100.0;
        let label = if pred_idx == 1 { "FACE" } else { "NO FACE" };
        
        Ok((label.to_string(), confidence))
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        std::process::exit(1);
    }
    
    println!("ResNet50 Face Classifier");
    
    let model = ResNet50FaceClassifier::new()?;
    
    println!("analyzing {}", args[1]);
    let (label, confidence) = model.predict(&args[1])?;

    if label == "FACE" {
        println!("FACE DETECTED");
    } else {
        println!("NO FACE DETECTED");
    }
    println!("Confidence: {:.2}%", confidence);

    Ok(())
}