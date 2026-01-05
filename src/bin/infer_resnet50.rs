use anyhow::{Result, Context};
use tch::{nn, nn::ModuleT, Tensor, Device, Kind, CModule};
use std::env;

struct ResNet50FaceClassifier {
    backbone: nn::VarStore,
    fc_module: CModule,
    device: Device,
}

impl ResNet50FaceClassifier {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        let mut backbone = nn::VarStore::new(device);
        
        // Load ResNet50 backbone
        backbone.load_partial("models/resnet50/resnet50.ot")
            .context("Failed to load ResNet50 backbone")?;
        
        println!("loaded ResNet50 backbone");
        
        // Load TorchScript FC module
        let fc_module = CModule::load("models/resnet50/fc_layer_resnet50_cpu.pt")
            .context("failed to load FC layer")?;
        
        println!("successfully loaded trained FC layer");
        
        Ok(ResNet50FaceClassifier { backbone, fc_module, device })
    }
    
    fn predict(&self, image_path: &str) -> Result<(String, f32)> {
        //load  + process
        let img = image::open(image_path)
            .context("Failed to open image")?
            .resize_exact(224, 224, image::imageops::FilterType::Lanczos3)
            .to_rgb8();
        
        //convert to tensor
        let pixels: Vec<f32> = img.pixels()
            .flat_map(|p| vec![p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        
        let tensor = Tensor::from_slice(&pixels)
            .view([224, 224, 3])
            .permute(&[2, 0, 1])
            / 255.0;
        
        // normalization ImageNet
        let mean = Tensor::from_slice(&[0.485_f32, 0.456, 0.406]).view([3, 1, 1]);
        let std = Tensor::from_slice(&[0.229_f32, 0.224, 0.225]).view([3, 1, 1]);
        let normalized = (tensor - mean) / std;
        
        // extract features
        let features = tch::no_grad(|| {
            let batch = normalized.unsqueeze(0);
            tch::vision::resnet::resnet50_no_final_layer(&self.backbone.root())
                .forward_t(&batch, false)
        });
        
        // run through FC layer
        let output = tch::no_grad(|| {
            self.fc_module.forward_ts(&[features]).unwrap()
        });
        
        // predict
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
        eprintln!("Example: bash scripts/run_resnet50.sh photo.jpg");
        std::process::exit(1);
    }
    
    println!("ResNet50 Face Classifier (99.97% Acc) ");
  
    
    let model = ResNet50FaceClassifier::new()?;
    
    println!("\nanalyzing {}\n", args[1]);
    let (label, confidence) = model.predict(&args[1])?;

    if label == "FACE" {
        println!("FACE DETECTED");
    } else {
        println!("NO FACE DETECTED");
    }
    println!("Confidence: {:.2}%", confidence);

    Ok(())
}